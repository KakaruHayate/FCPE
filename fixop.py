import torchaudio
import torch
import numpy as np
from fixsli import split_into_sentences
import os
import tqdm

if __name__ == '__main__':
    roodir = './op/train'
    filelist = os.listdir(os.path.join(roodir, 'raudio', '1'))
    for filename in tqdm.tqdm(filelist):
        filename = filename.split('.')[0]
        audio, sr = torchaudio.load(os.path.join(roodir, 'raudio', '1', filename + '.wav'))
        mask = np.load(os.path.join(roodir, 'rmask', '1', filename + '.npy'))
        ope = np.load(os.path.join(roodir, 'rope', '1', filename + '.npy'))
        assert len(mask) == len(ope)
        ope = ope * (~mask)
        sentences = split_into_sentences(mask, 100)
        for i in range(len(sentences)):
            j=0
            sentence = sentences[i]
            star, end = sentence
            if end - star < 10:
                continue
            if end - star > 1000:
                new_list = []
                k=(end - star)//500
                for kj in range(k):
                    if kj == 0:
                        new_list.append([star, star+(kj+1)*500])
                    elif kj == k-1:
                        new_list.append([star+kj*500 + 1, end])
                    else:
                        new_list.append([star+kj*500 + 1, star+(kj+1)*500])
                for sentence in new_list:
                    star, end = sentence
                    star = star - 20
                    end = end + 20
                    if star < 0:
                        star = 0
                    if end >= len(mask):
                        end = len(mask) - 1
                    end = end + 1
                    opes = ope[star:end]
                    audios = audio[:, star * 160:end * 160]
                    masks = mask[star:end]
                    torchaudio.save(os.path.join(roodir, 'audio', '1', filename + '_' + str(i) + '_' + str(j) + '.wav'),
                                    audios, sr)
                    np.save(os.path.join(roodir, 'ope', '1', filename + '_' + str(i) + '_' + str(j) + '.npy'), opes)
                    np.save(os.path.join(roodir, 'mask', '1', filename + '_' + str(i) + '_' + str(j) + '.npy'), masks)
                    j=j+1

            else:
                star = star-20
                end = end+20
                if star < 0:
                    star = 0
                if end >= len(mask):
                    end = len(mask) - 1
                end = end + 1
                opes = ope[star:end]
                audios = audio[:, star*160:end*160]
                masks = mask[star:end]
                torchaudio.save(os.path.join(roodir, 'audio', '1', filename + '_' + str(i) + '_' + str(j) + '.wav'), audios, sr)
                np.save(os.path.join(roodir, 'ope', '1', filename + '_' + str(i) + '_' + str(j) + '.npy'), opes)
                np.save(os.path.join(roodir, 'mask', '1', filename + '_' + str(i)  + '_'+ str(j) + '.npy'), masks)





