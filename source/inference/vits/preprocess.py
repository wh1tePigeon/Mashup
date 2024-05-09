import os
import sys
import numpy as np
import torch
import librosa
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.model.hubert.hubert import hubert_soft
from source.utils.pitch import compute_f0_sing, save_csv_pitch
import whisper


def get_ppg(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    ppg_save_path = os.path.join(directory_save_file, (filename + ".ppg.npy"))

    m = cfg["model"]
    if m not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
                    "tiny.en", "base.en", "small.en", "medium.en"]:
        raise KeyError
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(name=m, download_root=cfg["checkpoint_path"]).to(device)
    model.eval()

    audio = whisper.load_audio(filepath)
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 15 * 16000 < audln):
        short = audio[idx_s:idx_s + 15 * 16000]
        short = whisper.pad_or_trim(short)
        idx_s = idx_s + 15 * 16000
        ppgln = 15 * 16000 // 320
        mel = whisper.log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            mel = mel.unsqueeze(0)
            ppg = model.embed_audio(mel).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        short = whisper.pad_or_trim(short)
        mel = whisper.log_mel_spectrogram(short).to(device)
        if not (device == "cpu"):
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = model.embed_audio(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1024]
            ppg_a.extend(ppg)
    np.save(ppg_save_path, ppg_a, allow_pickle=False)

    return ppg_save_path


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


#if __name__ == "__main__":
    # cfg = {
    #     "filepath" : "/home/comp/Рабочий стол/Mashup/input/ramm_test.wav",
    #     "output_dir" : "/home/comp/Рабочий стол/Mashup/output/hubert",
    #     "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/hubert/hubert-soft-0d54a1f4.pt"
    # }

    # cfg = {
    #     "filepath" : "/home/comp/Рабочий стол/Mashup/input/ramm_test.wav",
    #     "output_dir" : "/home/comp/Рабочий стол/Mashup/output/pitch"
    # }

    # cfg = {
    #     "filepath" : "/home/comp/Рабочий стол/Mashup/input/ramm_test_vocal_short.wav",
    #     "output_dir" : "/home/comp/Рабочий стол/Mashup/output/whisper",
    #     "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/whisper",
    #     "model" : "small"
    # }
    # get_ppg(cfg)
