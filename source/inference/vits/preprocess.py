import os
import sys
import numpy as np
import argparse
import torch
import librosa
import torch
import torchaudio
import requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.model.hubert.hubert import hubert_soft
from source.utils.pitch import compute_f0_sing, save_csv_pitch

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

    hubert = hubert_soft(cfg["checkpoint_path"])
    audio, _ = librosa.load(filepath, sr=16000)
    audio = torch.from_numpy(audio).to(device)
    audio = audio[None, None, :]
    with torch.inference_mode():
        units = hubert.units(audio).squeeze().data.cpu().float().numpy()
        np.save(vec_save_path, units, allow_pickle=False)

    return vec_save_path


def get_pitch(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    pitch_save_path = os.path.join(directory_save_file, (filename + "_pitch.csv"))

    pitch = compute_f0_sing(filepath, device)
    save_csv_pitch(pitch, pitch_save_path)

    return pitch_save_path



if __name__ == "__main__":
    # cfg = {
    #     "filepath" : "/home/comp/Рабочий стол/Mashup/input/ramm_test.wav",
    #     "output_dir" : "/home/comp/Рабочий стол/Mashup/output/hubert",
    #     "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/hubert/hubert-soft-0d54a1f4.pt"
    # }

    cfg = {
        "filepath" : "/home/comp/Рабочий стол/Mashup/input/ramm_test.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/pitch"
    }
    get_pitch(cfg)