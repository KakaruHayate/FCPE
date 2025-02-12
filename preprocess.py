import os
import numpy as np
import librosa
import torch
import argparse
import shutil
from train.savertools import utils
from tqdm import tqdm
from preprocess.audio2mel import Audio2Mel
from preprocess.mask_util import MaskUtil
from librosa.filters import mel as librosa_mel_fn
from scipy.interpolate import interp1d
from train.savertools.utils import traverse_dir
import concurrent.futures
import csv

from icecream import ic

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

def read_csv_file(csv_file):
    time = []
    jaw_open = []
    mouth_close = None
    lips_distance = None
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        has_mouth_close = 'mouthClose' in columns
        has_lips_distance = 'LipsDistance' in columns
        for row in reader:
            time.append(float(row['TimeStamp']))
            jaw_open.append(float(row['jawOpen']))
            if has_mouth_close:
                if mouth_close is None:
                    mouth_close = []
                mouth_close.append(float(row['mouthClose']))
            if has_lips_distance:
                if lips_distance is None:
                    lips_distance = []
                lips_distance.append(float(row['LipsDistance']))
        if not has_mouth_close:
            print(f"Warning: 'mouthClose' column not found in {csv_file}. Returning None for mouth_close.")
        if not has_lips_distance:
            print(f"Warning: 'LipsDistance' column not found in {csv_file}. Returning None for lips_distance.")
    return np.array(time), np.array(jaw_open), np.array(mouth_close) if mouth_close is not None else None, np.array(lips_distance) if lips_distance is not None else None

def preprocess(
        fbl_onnx_path, 
        path_audiodir, 
        path_csvdir, 
        path_meldir, 
        path_opedir,
        path_maskdir, 
        path_skipdir,
        device,
        sampling_rate,
        hop_length,
        win_length,
        n_fft,
        n_mel_channels,
        mel_fmin,
        mel_fmax, 
        ope_mode, 
        mask_breath):
        
    # list files
    audio_filelist =  traverse_dir(
        path_audiodir,
        extension='wav',
        is_pure=True,
        is_sort=True,
        is_ext=True)

    csv_filelist =  traverse_dir(
        path_csvdir,
        extension='csv',
        is_pure=True,
        is_sort=True,
        is_ext=True)

    # initilize extractor
    mel_extractor = Audio2Mel(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        n_mel_channels=n_mel_channels,
        win_length=win_length,
        n_fft=n_fft,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
        clamp=1e-6).to(device)

    mask_gen = MaskUtil(
        fbl_onnx_path=fbl_onnx_path, 
        hop_size=hop_length, 
        sample_rate=sampling_rate, 
        mask_breath=mask_breath
        )

    # run
    
    def process(audio_file, csv_file):
        ext = audio_file.split('.')[-1]
        binfile = audio_file[:-(len(ext)+1)]+'.npy'
        path_audiofile = os.path.join(path_audiodir, audio_file)
        path_csvfile = os.path.join(path_csvdir, csv_file)
        path_melfile = os.path.join(path_meldir, binfile)
        path_opefile = os.path.join(path_opedir, binfile)
        path_maskfile = os.path.join(path_maskdir, binfile)
        
        # load audio
        x, _ = librosa.load(path_audiofile, sr=sampling_rate)
        x_t = torch.from_numpy(x).float().to(device)
        x_t = x_t.unsqueeze(0).unsqueeze(0) # (T,) --> (1, 1, T)

        # extract mel
        m_t = mel_extractor(x_t)
        mel = m_t.squeeze().to('cpu').numpy()
        t_mel = librosa.frames_to_time(np.arange(mel.shape[0]), sr=sampling_rate, hop_length=hop_length)
        
        # extract ope
        timestamps, jaw_open, mouth_close, lips_distance = read_csv_file(path_csvfile)
        if ope_mode=='mode2' or ope_mode=='mode3' and mouth_close is None:
            print('\n[Error] OPE extraction failed, mouthClose is None!!!: ' + path_csvfile)
            os.makedirs(path_skipdir, exist_ok=True)
            shutil.move(path_csvfile, path_skipdir)
            shutil.move(path_audiofile, path_skipdir)
        else:
            if ope_mode=='mode1':
                ope = jaw_open
            elif ope_mode=='mode2':
                ope = jaw_open - mouth_close
            elif ope_mode=='mode3':
                ope = jaw_open * (1-mouth_close)
            else:
                raise ValueError(f" [x] Unknown OPE mode: {ope_mode}")
            ope_interp_func = interp1d(
                timestamps, 
                ope, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(ope[0], ope[-1])
            )
            ope_aligned = ope_interp_func(t_mel)
            ope_aligned = np.nan_to_num(ope_aligned, nan=ope[-1])
            
            #build mask
            mask = mask_gen.build_mask(path_audiofile, mel.shape[0], timestamps, jaw_open, mouth_close, lips_distance)
            
            # save npy
            os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
            np.save(path_melfile, mel)
            os.makedirs(os.path.dirname(path_opefile), exist_ok=True)
            np.save(path_opefile, ope_aligned)
            os.makedirs(os.path.dirname(path_maskfile), exist_ok=True)
            np.save(path_maskfile, mask)        
    print('Preprocess the audio clips in :', path_audiodir)
    
    audio_filelist.sort()
    csv_filelist.sort()
    combined_filelist = zip(audio_filelist, csv_filelist)
    if len(audio_filelist) != len(csv_filelist):
        raise ValueError("The length of audio_filelist and csv_filelist must be the same.")
    # single process
    for audio_file, csv_file in tqdm(combined_filelist, total=len(audio_filelist)):
        process(audio_file, csv_file)
    
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
                
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    fbl_onnx_path = args.data.fbl_onnx_path
    sampling_rate  = args.mel.sampling_rate
    hop_length = args.mel.hop_size
    win_length = args.mel.win_size
    n_fft = args.mel.n_fft
    n_mel_channels = args.mel.num_mels
    mel_fmin = args.mel.fmin
    mel_fmax = args.mel.fmax
    train_path = args.data.train_path
    valid_path = args.data.valid_path
    ope_mode = args.data.ope_mode # 1.'jawOpen' 2.'jawOpen - mouthClose' 3.'jawOpen * (1 - mouthClose)'
    mask_breath = args.data.mask_breath # 注意这里如果是False，并不是跳过换气声检测，而是从vad结果中去掉换气声部分
    
    # run
    for path in [train_path, valid_path]:
        path_audiodir  = os.path.join(path, 'audio')
        path_csvdir  = os.path.join(path, 'csv')
        path_meldir  = os.path.join(path, 'mel')
        path_opedir  = os.path.join(path, 'ope')
        path_maskdir  = os.path.join(path, 'mask')
        path_skipdir = os.path.join(path, 'skip')
        preprocess(
            fbl_onnx_path, 
            path_audiodir, 
            path_csvdir, 
            path_meldir, 
            path_opedir,
            path_maskdir, 
            path_skipdir,
            device,
            sampling_rate,
            hop_length,
            win_length,
            n_fft,
            n_mel_channels,
            mel_fmin,
            mel_fmax, 
            ope_mode, 
            mask_breath)
    
