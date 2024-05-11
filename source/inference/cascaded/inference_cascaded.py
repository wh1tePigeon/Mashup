import os
import sys
import torch
import torchaudio as ta
import librosa
import numpy as np
from hydra.utils import instantiate
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device
from omegaconf import OmegaConf
from source.model.cascaded.cascaded import CascadedNet
from source.model.cascaded.separator import Separator
from source.utils.spec_utils import wave_to_spectrogram, spectrogram_to_wave

def inference_cascaded(cfg):
    # device, device_ids = prepare_device(cfg["n_gpu"])
    # arch = OmegaConf.load(cfg["model"])
    # model = instantiate(arch)
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    # separator = 

    # checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    # state_dict = checkpoint["state_dict"]
    # model.load_state_dict(state_dict)
    # model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    model = CascadedNet(cfg["n_fft"], cfg["hop_length"], 32, 128)
    model.to(device)
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt)
    

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]

    if os.path.isfile(filepath):
        sr = 44100
        audio, _ = librosa.load(
        filepath, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        filename = filepath.split(".")[0].split("/")[-1]
        
        directory_save_file = os.path.join(output_dir, filename)
        os.makedirs(directory_save_file, exist_ok=True)
        
        vocal_save_path = os.path.join(directory_save_file, (filename + "_vocal.wav"))
        background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))
        
        if audio.ndim == 1:
        # mono to stereo
            audio = np.asarray([audio, audio])

        audio_spec = wave_to_spectrogram(audio, cfg["hop_length"], cfg["n_fft"])

        sp = Separator(
            model=model,
            device=device,
            batchsize=cfg["batchsize"],
            cropsize=cfg["cropsize"],
            postprocess=cfg["postprocess"]
        )

        print("Separating " + filename)
        background_spec, vocal_spec = sp.separate(audio_spec)

        background = torch.from_numpy(spectrogram_to_wave(background_spec, hop_length=cfg["hop_length"]))
        vocal = torch.from_numpy(spectrogram_to_wave(vocal_spec, hop_length=cfg["hop_length"]))

        if cfg["save"]:
            ta.save(vocal_save_path, vocal, sample_rate=sr)
            ta.save(background_save_path, background, sample_rate=sr)

            return [vocal_save_path, background_save_path, sr]
        else:
            return vocal, background, sr


if __name__ == "__main__":
    cfg = {
        "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/cascaded/baseline.pth",
        "filepath" : "/home/comp/Рабочий стол/Mashup/input/test_2.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output",
        "hop_length" : 1024,
        "n_fft" : 2048,
        "batchsize" : 4,
        "cropsize" : 256,
        "postprocess" : False
    }
    inference_cascaded(cfg)