import os
import shutil
import argparse
import torchaudio
from train.savertools import utils
from tqdm import tqdm

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

def resample_audio(input_path, output_path, new_sample_rate):
    waveform, sample_rate = torchaudio.load(input_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resampler(waveform)
    torchaudio.save(output_path, resampled_waveform, new_sample_rate)

def process_files(source_dir, audio_dest_dir, csv_dest_dir, new_sample_rate=16000):
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    for subdir in tqdm(subdirs, desc="Processing folders"):
        subdir_path = os.path.join(source_dir, subdir)
        audio_file = os.path.join(subdir_path, 'audio.wav')
        csv_file = os.path.join(subdir_path, 'mouth_data.csv')

        # Check if both audio and CSV files exist
        if os.path.exists(audio_file):
            if os.path.exists(csv_file):
                # Copy and resample audio file
                audio_output_path = os.path.join(audio_dest_dir, f'{subdir}.wav')
                resample_audio(audio_file, audio_output_path, new_sample_rate)

                # Copy CSV file
                csv_output_path = os.path.join(csv_dest_dir, f'{subdir}.csv')
                shutil.copy(csv_file, csv_output_path)
            else:
                print(f'{csv_file} does not exist. Skipping folder {subdir}.')
        else:
            print(f'{audio_file} does not exist. Skipping folder {subdir}.')

if __name__ == '__main__':
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    new_sample_rate = args.data.sampling_rate
    source_directory = './op/raw/'
    audio_destination_directory = './op/train/1/audio/'
    csv_destination_directory = './op/train/1/csv/'

    # Ensure destination directories exist
    os.makedirs(audio_destination_directory, exist_ok=True)
    os.makedirs(csv_destination_directory, exist_ok=True)

    process_files(source_directory, audio_destination_directory, csv_destination_directory, new_sample_rate)
