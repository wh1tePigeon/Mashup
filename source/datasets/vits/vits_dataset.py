import os
import numpy as np
import random
import torch
import torch.utils.data
import torchaudio as ta
from scipy.io.wavfile import read


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


class TextAudioSpeakerSet(torch.utils.data.Dataset):
    def __init__(self, filename, hparams):
        self.items = load_filepaths(filename)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.segment_size = hparams.segment_size
        self.hop_length = hparams.hop_length
        self._filter()
        print(f'----------{len(self.items)}----------')

    def _filter(self):
        lengths = []
        items_new = []
        items_min = int(self.segment_size / self.hop_length * 4)  # 1 S
        items_max = int(self.segment_size / self.hop_length * 16)  # 4 S
        for wavpath, spec, pitch, vec, ppg, spk in self.items:
            if not os.path.isfile(wavpath):
                continue
            if not os.path.isfile(spec):
                continue
            if not os.path.isfile(pitch):
                continue
            if not os.path.isfile(vec):
                continue
            if not os.path.isfile(ppg):
                continue
            if not os.path.isfile(spk):
                continue
            temp = np.load(pitch)
            usel = int(temp.shape[0] - 1)  # useful length
            if (usel < items_min):
                continue
            if (usel >= items_max):
                usel = items_max
            items_new.append([wavpath, spec, pitch, vec, ppg, spk, usel])
            lengths.append(usel)
        self.items = items_new
        self.lengths = lengths

    # i spent ~12h debugging this, cause somehow nothing works with torchaudio ¯\_(ツ)_/¯
    # def read_wav(self, filename):
    #     audio, sampling_rate = ta.load(filename)
    #     assert sampling_rate == self.sampling_rate, f"error: this sample rate of {filename} is {sampling_rate}"
    #     audio_norm = audio / self.max_wav_value
    #     return audio_norm

    def read_wav(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        assert sampling_rate == self.sampling_rate, f"error: this sample rate of {filename} is {sampling_rate}"
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm


    def __getitem__(self, index):
        return self.my_getitem(index)

    def __len__(self):
        return len(self.items)

    def my_getitem(self, idx):
        item = self.items[idx]
        # print(item)
        wav = item[0]
        spe = item[1]
        pit = item[2]
        vec = item[3]
        ppg = item[4]
        spk = item[5]
        use = item[6]

        wav = self.read_wav(wav)
        spe = torch.load(spe)

        pit = np.load(pit)
        vec = np.load(vec)
        vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
        ppg = np.load(ppg)
        ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        spk = np.load(spk)

        pit = torch.FloatTensor(pit)
        vec = torch.FloatTensor(vec)
        ppg = torch.FloatTensor(ppg)
        spk = torch.FloatTensor(spk)

        len_pit = pit.size()[0]
        len_vec = vec.size()[0] - 2 # for safe
        len_ppg = ppg.size()[0] - 2 # for safe
        len_min = min(len_pit, len_vec)
        len_min = min(len_min, len_ppg)
        len_wav = len_min * self.hop_length

        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]
        spe = spe[:, :len_min]
        wav = wav[:, :len_wav]
        if len_min > use:
            max_frame_start = ppg.size(0) - use - 1
            frame_start = random.randint(0, max_frame_start)
            frame_end = frame_start + use

            pit = pit[frame_start:frame_end]
            vec = vec[frame_start:frame_end, :]
            ppg = ppg[frame_start:frame_end, :]
            spe = spe[:, frame_start:frame_end]

            wav_start = frame_start * self.hop_length
            wav_end = frame_end * self.hop_length
            wav = wav[:, wav_start:wav_end]
        # print(spe.shape)
        # print(wav.shape)
        # print(ppg.shape)
        # print(pit.shape)
        # print(spk.shape)
        return spe, wav, ppg, vec, pit, spk