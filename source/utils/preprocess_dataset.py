import torch
import torchaudio as ta
import torch.nn.functional as F
import os
import sys
from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm
import librosa
import pyworld
import numpy as np
import torchcrepe
from whisper import Whisper, ModelDimensions, pad_or_trim, log_mel_spectrogram
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.model.hubert.hubert import hubert_soft
import json
from TTS.tts.utils.speakers import SpeakerManager
from TTS.encoder.models.lstm import LSTMSpeakerEncoder
from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig
from TTS.utils.audio import AudioProcessor


def process_dir_pitch(speaker_dirpath, output_dir, use_pyworld=True, use_crepe=False):
    assert os.path.exists(speaker_dirpath)
    assert use_pyworld ^ use_crepe

    speakername = os.path.basename(os.path.normpath(speaker_dirpath))
    savepaths = []

    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Extracting pitch for speaker {speakername}'):
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
            savepaths.append(savepath)

    return savepaths


def process_dir_whisper(speaker_dirpath, output_dir, model, checkpoint_path):
    assert os.path.exists(speaker_dirpath)

    if model not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
                    "tiny.en", "base.en", "small.en", "medium.en"]:
        raise KeyError
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    if not (device == "cpu"):
        model.half()
    model.to(device)
    
    speakername = os.path.basename(os.path.normpath(speaker_dirpath))
    savepaths = []

    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Extracting ppg for speaker {speakername}'):
        if filename.endswith(".wav"):
            filepath = os.path.join(speaker_dirpath, filename)

            audio, sr = librosa.load(filepath, sr=16000)
            assert sr == 16000

            audln = audio.shape[0]
            ppgln = audln // 320
            audio = pad_or_trim(audio)
            mel = log_mel_spectrogram(audio).to(device)
            if not (device == "cpu"):
                mel = mel.half()
            with torch.no_grad():
                ppg = model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
                ppg = ppg[:ppgln,]  # [length, dim=1280]
                
                filename = filename[:-4]
                savepath = os.path.join(output_dir, (filename + "_ppg"))
                np.save(savepath, ppg, allow_pickle=False)
                savepaths.append(savepath)
    
    return savepaths



def process_dir_hubert(speaker_dirpath, output_dir, checkpoint_path):
    assert os.path.exists(speaker_dirpath)

    hubert = hubert_soft(checkpoint_path)
    savepaths = []
    speakername = os.path.basename(os.path.normpath(speaker_dirpath))

    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Extracting vec for speaker {speakername}'):
        if filename.endswith(".wav"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            filepath = os.path.join(speaker_dirpath, filename)

            audio, sr = librosa.load(filepath, sr=16000)
            assert sr == 16000

            audio = torch.from_numpy(audio).to(device)
            audio = audio[None, None, :]
            with torch.inference_mode():
                units = hubert.units(audio).squeeze().data.cpu().float().numpy()
                filename = filename[:-4]
                savepath = os.path.join(output_dir, (filename + "_vec"))
                np.save(savepath, units, allow_pickle=False)
                savepaths.append(savepath)

    return savepaths


def process_dir_spk(speaker_dirpath, output_dir, checkpoint_path, cfg_path):
    assert os.path.exists(speaker_dirpath)

    speakername = os.path.basename(os.path.normpath(speaker_dirpath))
    savepaths = []

    with open(cfg_path, 'r') as file:
        cfg_dict = json.load(file)

    config = SpeakerEncoderConfig(cfg_dict)
    config.from_dict(cfg_dict)

    speaker_encoder = LSTMSpeakerEncoder(
        config.model_params["input_dim"],
        config.model_params["proj_dim"],
        config.model_params["lstm_dim"],
        config.model_params["num_lstm_layers"],
    )
    speaker_encoder.load_checkpoint(config, checkpoint_path, eval=True, use_cuda=torch.cuda.is_available())

    ap = AudioProcessor(**cfg_dict["audio"])
    ap.do_sound_norm = True

    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Computing embds for speaker {speakername}'):
        if filename.endswith(".wav"):
            filepath = os.path.join(speaker_dirpath, filename)
            
            # waveform = ap.load_wav(filepath, sr=ap.sample_rate)
            # mel = ap.melspectrogram(waveform)
            # spec = torch.from_numpy(mel.T)
            # if torch.cuda.is_available():
            #     spec = spec.cuda()

            # spec = spec.unsqueeze(0)
            # embd = speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
            # embd = embd.squeeze()

            embd = np.array(0)

            filename = filename[:-4]
            savepath = os.path.join(output_dir, (filename + "_spk"))
            np.save(savepath, embd, allow_pickle=False)
            savepaths.append(savepath)

    return savepaths



def process_dir_spk_average(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)

    speakername = os.path.basename(os.path.normpath(speaker_dirpath))
    count = 0
    average = 0
    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Computing average embd for speaker {speakername}'):
        if filename.endswith(".npy"):
            filepath = os.path.join(speaker_dirpath, filename)

            source_embed = np.load(filepath)
            source_embed = source_embed.astype(np.float32)
            average = average + source_embed
            count = count + 1

    if count > 0:
        average = average / count
        savepath = os.path.join(output_dir, speakername)
        np.save(savepath, average, allow_pickle=False)
    
    return savepath
    

def process_dir_sr32k(speaker_dirpath, output_dir):
    assert os.path.exists(speaker_dirpath)

    speakername = os.path.basename(os.path.normpath(speaker_dirpath))

    savepaths = []

    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Resampling to 32 kHz for speaker {speakername}'):
        if filename.endswith(".wav"):
            filepath = os.path.join(speaker_dirpath, filename)

            audio, sr = librosa.load(filepath, sr=32000)
            audio = torch.tensor(np.copy(audio))[None]

            filename = filename[:-4]
            savepath = os.path.join(output_dir, (filename + "_sr32k.wav"))
            ta.save(savepath, audio, sr)
            savepaths.append(savepath)

    return savepaths

    
def process_dir_specs(speaker_dirpath, output_dir, spec_cfg):
    assert os.path.exists(speaker_dirpath)

    speakername = os.path.basename(os.path.normpath(speaker_dirpath))

    savepaths = []

    for filename in tqdm(os.listdir(speaker_dirpath), desc=f'Computing specs for speaker {speakername}'):
        if filename.endswith(".wav"):
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            filepath = os.path.join(speaker_dirpath, filename)

            audio, _ = librosa.load(filepath, sr=spec_cfg["sr"])
            audio = torch.tensor(np.copy(audio))[None]
            audio_norm = audio / spec_cfg["max_wav_value"]
            n_fft = spec_cfg["filter_length"]
            hop_size = spec_cfg["hop_length"]
            win_size = spec_cfg["win_length"]

            hann_window = torch.hann_window(win_size).to(dtype=audio_norm.dtype, device=audio_norm.device)

            audio_norm = torch.nn.functional.pad(
                audio_norm.unsqueeze(1),
                (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                mode="reflect",
            )
            audio_norm = audio_norm.squeeze(1)

            spec = torch.stft(
                audio_norm,
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=hann_window,
                center=False,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=False,
            )

            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
            spec = torch.squeeze(spec, 0)

            filename = filename[:-4]
            savepath = os.path.join(output_dir, (filename + "_spec.pt"))
            torch.save(spec, savepath)
            savepaths.append(savepath)
    
    return savepaths
    

def process_dir(dirpath, output_dir, model_cfgs):
    assert os.path.exists(dirpath)

    pitches_path = os.path.join(output_dir, "pitch")
    whispers_path = os.path.join(output_dir, "whisper")
    huberts_path = os.path.join(output_dir, "hubert")
    specs_path = os.path.join(output_dir, "specs")
    spks_path = os.path.join(output_dir, "spks")
    spks_average = os.path.join(output_dir, "spks_average")
    sr32k_paths = os.path.join(output_dir, "sr32k")

    common = []
    
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

            pitch_spk_paths = process_dir_pitch(speakerdir, pit_savepath)
            whisper_spk_paths = process_dir_whisper(speakerdir, whisper_savepath, **model_cfgs["whisper"])
            hubert_spk_paths = process_dir_hubert(speakerdir, hubert_savepath, **model_cfgs["hubert"])
            embds_spk_paths = process_dir_spk(speakerdir, spk_savepath, **model_cfgs["spk_enc"])
            embds_spk_average_path = process_dir_spk_average(spk_savepath, spk_average_savepath)
            sr32k_spk_paths = process_dir_sr32k(speakerdir, sr32k_savepaths)
            specs_spk_paths = process_dir_specs(sr32k_savepaths, spec_savepath, model_cfgs["spec"])

            for i in range(0, len(sr32k_spk_paths)):
                common.append(
                        f"{sr32k_spk_paths[i]}|{specs_spk_paths[i]}| \
                            {pitch_spk_paths[i]}|{hubert_spk_paths[i]}|{whisper_spk_paths[i]}|{embds_spk_paths[i]}")

    random.shuffle(common)
    split = int(len(common) * 0.2)
    valids = common[:split]
    valids.sort()
    trains = common[split:]
    trains.sort()

    valids_savepath = os.path.join(output_dir, "valid.txt")
    trains_savepath = os.path.join(output_dir, "train.txt")

    fw = open(valids_savepath, "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open(trains_savepath, "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()

    return [valids_savepath, trains_savepath]

if __name__ == "__main__":
    cfg = {
        "dirpath" : "/home/comp/Рабочий стол/dataset/",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/dataset",
        "model_cfgs" : {
            "whisper": {
                "model" : "small",
                "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/whisper/small.pt"
            },
            "hubert" : {
                "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/hubert/hubert-soft-0d54a1f4.pt"
            },
            "spec" : {
                "sr" : 32000,
                "max_wav_value" : 32768.0,
                "filter_length" : 1024,
                "hop_length" : 320,
                "win_length" : 1024 
            },
            "spk_enc" : {
                "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/speaker/best_model.pth.tar",
                "cfg_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/speaker/cfg_nc.json"
            }
        }
    }

    process_dir(**cfg)