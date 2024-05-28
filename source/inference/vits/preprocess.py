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
from whisper import Whisper, ModelDimensions, pad_or_trim, log_mel_spectrogram


def get_ppg(cfg):
    print("extracting ppg")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    m = cfg["model"]
    if m not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
                    "tiny.en", "base.en", "small.en", "medium.en"]:
        raise KeyError
    
    checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    # if not (device == "cpu"):
    #     model.half()
    model.to(device)

    audio, sr = librosa.load(filepath, sr=16000)
    assert sr == 16000

    audln = audio.shape[0]
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(device)
    # if not (device == "cpu"):
    #     mel = mel.half()
    with torch.no_grad():
        ppg = model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,]  # [length, dim=1280]

        savepath = os.path.join(directory_save_file, (filename + "_ppg"))
        np.save(savepath, ppg, allow_pickle=False)


    return savepath


def get_vec(cfg):
    print("extracting vec")
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    vec_save_path = os.path.join(directory_save_file, (filename + "_hubert.npy"))

    hubert = hubert_soft(cfg["checkpoint_path"]).to(device)
    audio, _ = librosa.load(filepath, sr=16000)
    audio = torch.from_numpy(audio).to(device)
    audio = audio[None, None, :]
    with torch.inference_mode():
        units = hubert.units(audio).squeeze().data.cpu().float().numpy()
        np.save(vec_save_path, units, allow_pickle=False)

    return vec_save_path


def get_pitch(cfg):
    print("extracting pitch")
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
    cfg1 = {
        "filepath" : "/home/comp/Рабочий стол/samples/govnovoz_vocal.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/hubert",
        "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/hubert/hubert-soft-0d54a1f4.pt"
    }

    cfg2 = {
        "filepath" : "/home/comp/Рабочий стол/samples/govnovoz_vocal.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/pitch"
    }

    cfg3 = {
        "filepath" : "/home/comp/Рабочий стол/samples/govnovoz_vocal.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/whisper",
        "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/whisper/small.pt",
        "model" : "small"
    }
    get_vec(cfg1)
    get_pitch(cfg2)
    get_ppg(cfg3)
