import os
import sys
import demucs.api
import torchaudio as ta
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.inference.cascaded.inference_cascaded import inference_cascaded
from source.inference.bsrnn.inference_bsrnn import inference_bsrnn


def separate_with_cascaded(filepath, model_cfg):
    model_cfg["filepath"] = filepath
    return inference_cascaded(model_cfg)


def separate_with_bsrnn_music(filepath, model_cfg):
    raise NotImplementedError()


def separate_with_bsrnn_speech(filepath, model_cfg):
    model_cfg["filepath"] = filepath
    return inference_bsrnn(model_cfg)


def separate_with_hybdemucs(filepath, model_cfg):
    #model_cfg["filepath"] = filepath
    audio, sr = ta.load(filepath)
    if sr != 44100:
        audio = ta.functional.resample(audio, sr, 44100)
        sr = 44100
    #separator = demucs.api.Separator(repo=Path(model_cfg["checkpoint_path"]))
    separator = demucs.api.Separator()
    origin, separated = separator.separate_tensor(audio)
    vocal = separated["vocals"]
    background = origin - vocal

    return vocal, background, sr


def separate(dirpath: str, ouput_dir: str, model_type: str, model_cfg: dict):
    assert os.path.exists(dirpath)
    model_types = ["cascaded", "bsrnn_music", "bsrnn_speech", "hybdemucs"]
    if model_type not in model_types:
        raise KeyError
    
    ouput_dir = os.path.join(ouput_dir, model_type)
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
            elif model_type == "hybdemucs":
                voice, background, sr = separate_with_hybdemucs(filepath, model_cfg)
            
            voice_save_path = os.path.join(voice_dir, filename + "_voice.wav")
            background_save_path = os.path.join(background_dir, filename + "_background.wav")

            separated_voice_filepaths.append(voice_save_path)
            separated_background_filepaths.append(background_save_path)

            print("Saving " + filename + "_voice.wav")
            print("Saving " + filename + "_background.wav")

            ta.save(voice_save_path, voice, sr)
            ta.save(background_save_path, background, sr)

    df = pd.DataFrame({'original_path': original_filepaths,
                    'separated_voice_path': separated_voice_filepaths,
                    'separated_background_path': separated_background_filepaths})

    csv_save_path = os.path.join(ouput_dir, "paths.csv")
    df.to_csv(csv_save_path, index=True)

    return csv_save_path


if __name__ == "__main__":
    cfg_cascaded = {
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

    cfg_hybdemucs = {
        "dirpath" : "/home/comp/Рабочий стол/denoise",
        "ouput_dir" : "/home/comp/Рабочий стол/denoised",
        "model_type" : "hybdemucs",
        "model_cfg" :  {
            "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/htdemucs",
            "filepath" : "/home/comp/Рабочий стол/Mashup/input/test_2.wav"
    }
    }

    cfg_bsrnn_speech = {
        "dirpath" : "/home/comp/Рабочий стол/denoise",
        "ouput_dir" : "/home/comp/Рабочий стол/denoised",
        "model_type" : "bsrnn_speech",
        "model_cfg" : {
            "save" : False,
            'n_gpu': 1,
            'model': "/home/comp/Рабочий стол/Mashup/source/configs/bsrnn/arch/model_conf.yaml",
            'sr': 44100,
            'filepath': {},
            'output_dir': "/home/comp/Рабочий стол/Mashup/output/bsrnn",
            'checkpoint_path': "/home/comp/Рабочий стол/Mashup/checkpoints/bsrnn/main.pth",
            'window_type': "hann",
            'chunk_size_second': 6.0,
            'hop_size_second': 0.5,
            'batch_size': 6
}

    }

    separate(**cfg_bsrnn_speech)