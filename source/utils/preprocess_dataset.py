import torch
import torchaudio as ta
import torch.nn.functional as F
import os
from typing import Tuple
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm
import librosa
import pyworld
import numpy as np
import torchcrepe
from whisper import Whisper, ModelDimensions


def process_dir_pitch(speaker_dirpath, output_dir, use_pyworld=True, use_crepe=False):
    assert os.path.exists(speaker_dirpath)
    assert use_pyworld ^ use_crepe

    for filename in os.listdir(speaker_dirpath):
        if filename.endswith(".wav"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            filepath = os.path.join(speaker_dirpath, filename)

            audio, sr = librosa.load(filepath, sr=16000)
            assert sr == 16000

            if use_pyworld:
                f0, t = pyworld.dio(
                    audio.astype(np.double),
                    fs=sr,
                    f0_ceil=900,
                    frame_period=1000 * 160 / sr,
                )
                f0 = pyworld.stonemask(audio.astype(np.double), f0, t, fs=16000)
                for index, pitch in enumerate(f0):
                    f0[index] = round(pitch, 1)

            #slow af
            if use_crepe:
                audio_torch = torch.tensor(np.copy(audio))[None]
                audio_torch = audio_torch + torch.randn_like(audio) * 0.001
                # Here we'll use a 20 millisecond hop length
                hop_length = 320
                fmin = 50
                fmax = 1000
                model = "full"
                batch_size = 512
                pitch = torchcrepe.predict(
                    audio_torch,
                    sr,
                    hop_length,
                    fmin,
                    fmax,
                    model,
                    batch_size=batch_size,
                    device=device,
                    return_periodicity=False,
                )
                pitch = np.repeat(pitch, 2, -1)  # 320 -> 160 * 2
                pitch = torchcrepe.filter.mean(pitch, 5)
                pitch = pitch.squeeze(0)

                f0 = pitch.numpy()

            filename = filename[:-4]
            savepath = os.path.join(output_dir, (filename + "_pitch"))
            np.save(savepath, f0, allow_pickle=False)



def process_dir_whisper(speaker_dirpath, output_dir, model, checkpoint_path):
    assert os.path.exists(speaker_dirpath)

    if model not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
                    "tiny.en", "base.en", "small.en", "medium.en"]:
        raise KeyError
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(dims)
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.half()
    model.to(device)
    
    for filename in os.listdir(speaker_dirpath):
        if filename.endswith(".wav"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            filepath = os.path.join(speaker_dirpath, filename)

            audio, sr = librosa.load(filepath, sr=16000)
            assert sr == 16000




def process_dir_hubert(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)
    
def process_dir_spk(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)

def process_dir_spk_average(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)
    
def process_dir_sr32k(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)
    
def process_dir_specs(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)
    

def process_dir(dirpath, output_dir):
    assert os.path.exists(dirpath)

    pitches_path = os.path.join(output_dir, "pitch")
    whispers_path = os.path.join(output_dir, "whisper")
    huberts_path = os.path.join(output_dir, "hubert")
    specs_path = os.path.join(output_dir, "specs")
    spks_path = os.path.join(output_dir, "spks")
    spks_average = os.path.join(output_dir, "spks_average")
    sr32k_paths = os.path.join(output_dir, "sr32k")
    
    for speakername in os.listdir(dirpath):
        speakerdir = os.path.join(dirpath, speakername)
        if os.path.isdir(speakerdir):
            pit_savepath = os.path.join(pitches_path, speakername)
            whisper_savepath = os.path.join(whispers_path, speakername)
            hubert_savepath = os.path.join(huberts_path, speakername)
            spec_savepath = os.path.join(specs_path, speakername)
            spk_savepath = os.path.join(spks_path, speakername)
            spk_average_savepath = os.path.join(spks_average, speakername)
            sr32k_savepaths = os.path.join(sr32k_paths, speakername)

            os.makedirs(pit_savepath, exist_ok=True)
            os.makedirs(whisper_savepath, exist_ok=True)
            os.makedirs(hubert_savepath, exist_ok=True)
            os.makedirs(spec_savepath, exist_ok=True)
            os.makedirs(spk_savepath, exist_ok=True)
            os.makedirs(spk_average_savepath, exist_ok=True)
            os.makedirs(sr32k_savepaths, exist_ok=True)

            process_dir_pitch(speakerdir, pit_savepath)
            process_dir_whisper(speakerdir, whisper_savepath)
            process_dir_hubert(speakerdir, hubert_savepath)
            process_dir_spk(speakerdir, spk_savepath)
            process_dir_spk_average(spk_savepath, spk_average_savepath)
            process_dir_sr32k(speakerdir, sr32k_savepaths)
            process_dir_specs(speakerdir, spec_savepath)


            # for filename in os.listdir(speakerdir):
            #     if filename.endswith(".wav"):
            #         filepath = os.path.join(speakerdir, filename)
            #         filename = filename[:-4]








if __name__ == "__main__":
    cfg = {
        "dirpath" : "/home/comp/Рабочий стол/dataset/",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/dataset"
    }
    process_dir(**cfg)