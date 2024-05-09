import os
import sys
import numpy as np
import argparse
import torch
import librosa
import torch
import torchaudio
import requests

def get_ppg(cfg):
    return 0

def get_vec(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    vec_save_path = os.path.join(directory_save_file, (filename + "_hubert.npy"))

    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(device)

    audio, sr = torchaudio.load(filepath)
    audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio.to(device)

    with torch.inference_mode():
        units = hubert.units(audio).squeeze().data.cpu().float().numpy()
        np.save(vec_save_path, units, allow_pickle=False)

    return vec_save_path

def get_pitch(cfg):
    return 0

if __name__ == "__main__":
    cfg = {
        "filepath" : "/home/comp/Рабочий стол/Mashup/input/ramm_test.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/hubert"
    }
    get_vec(cfg)