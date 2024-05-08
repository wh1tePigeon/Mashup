import numpy as np
import argparse
import torch
import librosa
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


from source.model.hubert.hubert import hubert_soft


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def load_model(path, device):
    model = hubert_soft(path)
    model.eval()
    if not (device == "cpu"):
        model.half()
    model.to(device)
    return model


def pred_vec(model, wavPath, vecPath, device):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    vec_a = []
    idx_s = 0
    while (idx_s + 20 * 16000 < audln):
        feats = audio[idx_s:idx_s + 20 * 16000]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            vec_a.extend(vec)
        idx_s = idx_s + 20 * 16000
    if (idx_s < audln):
        feats = audio[idx_s:audln]
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :]
        if not (device == "cpu"):
            feats = feats.half()
        with torch.no_grad():
            vec = model.units(feats).squeeze().data.cpu().float().numpy()
            # print(vec.shape)   # [length, dim=256] hop=320
            vec_a.extend(vec)
    np.save(vecPath, vec_a, allow_pickle=False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    # parser.add_argument("-v", "--vec", help="vec", dest="vec", required=True)
    # args = parser.parse_args()
    # print(args.wav)
    # print(args.vec)

    # wavPath = args.wav
    # vecPath = args.vec

    wavPath = "/home/comp/Рабочий стол/Mashup/output/ramm_test/ramm_test_vocal.wav"
    vecPath = "/home/comp/Рабочий стол/Mashup/output/hubert"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert = load_model(os.path.join("/home/comp/Рабочий стол/Mashup/checkpoints/hubert", "hubert-soft-0d54a1f4.pt"), device)

    pred_vec(hubert, wavPath, vecPath, device)