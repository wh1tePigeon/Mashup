import os
import sys
import torch
import torchaudio as ta
import pandas as pd
from speechbrain.inference.VAD import VAD
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.utils.process_audio import load_n_process_audio
from source.inference.cascaded.inference_cascaded import inference_cascaded


def separate_with_cascaded(filepath, model_cfg):
    model_cfg["filepath"] = filepath
    return inference_cascaded(model_cfg)


def separate_with_bsrnn_music(filepath, model_cfg):
    raise NotImplementedError()


def separate_with_bsrnn_speech(filepath, model_cfg):
    raise NotImplementedError()


def separate_with_hybdemucs(filepath, model_cfg):
    raise NotImplementedError()


def separate(dirpath: str, ouput_dir: str, model_type: str, model_cfg: dict):
    assert os.path.exists(dirpath)
    model_types = ["cascaded", "bsrnn_music", "bsrnn_speech", "hybdemucs"]
    if model_type not in model_types:
        raise KeyError
    
    voice_dir = os.path.join(ouput_dir, "voice")
    background_dir = os.path.join(ouput_dir, "background")
    os.makedirs(voice_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    original_filepaths = []
    separated_voice_filepaths = []
    separated_background_filepaths = []

    for filename in os.listdir(dirpath):
        if filename.endswith(".wav"):
            filepath = os.path.join(dirpath, filename)
            original_filepaths.append(filepath)

            filename = filepath.split(".")[0].split("/")[-1]

            if model_type == "cascaded":
                voice, background, sr = separate_with_cascaded(filepath, model_cfg)
            elif model_type == "bsrnn_music":
                voice, background, sr = separate_with_bsrnn_music(filepath, model_cfg)
            elif model_type == "bsrnn_speech":
                voice, background, sr = separate_with_bsrnn_speech(filepath, model_cfg)
            elif model_type == "bsrnn_music":
                voice, background, sr = separate_with_hybdemucs(filepath, model_cfg)
            
            voice_save_path = os.path.join(voice_dir, filename + "_voice.wav")
            background_save_path = os.path.join(background_dir, filename + "_background.wav")

            separated_voice_filepaths.append(voice_save_path)
            separated_background_filepaths.append(background_save_path)

            ta.save(voice_save_path, voice, sr)
            ta.save(background_save_path, background, sr)

    df = pd.DataFrame({'original_path': original_filepaths,
                    'separated_voice_path': separated_voice_filepaths,
                    'separated_background_path': separated_background_filepaths})

    csv_save_path = os.path.join(ouput_dir, "paths.csv")
    df.to_csv(csv_save_path, index=True)

    return csv_save_path


if __name__ == "__main__":
    cfg = {
        "dirpath" : "/home/comp/Рабочий стол/denoise",
        "ouput_dir" : "/home/comp/Рабочий стол/denoised",
        "model_type" : "cascaded",
        "model_cfg" :  {
        "save" : False,
        "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/cascaded/baseline.pth",
        "filepath" : "/home/comp/Рабочий стол/Mashup/input/test_2.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output",
        "hop_length" : 1024,
        "n_fft" : 2048,
        "batchsize" : 4,
        "cropsize" : 256,
        "postprocess" : False
    }
    }
    separate(**cfg)