import torch
import numpy as np
import librosa
import torchcrepe


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * \
        np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
        (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (
        f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min(
    ) >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def compute_f0_voice(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 10 millisecond hop length
    hop_length = 160
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=False,
    )
    pitch = torchcrepe.filter.mean(pitch, 3)
    pitch = pitch.squeeze(0)
    return pitch


def compute_f0_sing(filename, device):
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 20 millisecond hop length
    hop_length = 320
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 512
    pitch = torchcrepe.predict(
        audio,
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
    return pitch


def save_csv_pitch(pitch, path):
    with open(path, "w", encoding='utf-8') as pitch_file:
        for i in range(len(pitch)):
            t = i * 10
            minute = t // 60000
            seconds = (t - minute * 60000) // 1000
            millisecond = t % 1000
            print(
                f"{minute}m {seconds}s {millisecond:3d},{int(pitch[i])}", file=pitch_file)


def load_csv_pitch(path):
    pitch = []
    with open(path, "r", encoding='utf-8') as pitch_file:
        for line in pitch_file.readlines():
            pit = line.strip().split(",")[-1]
            pitch.append(int(pit))
    return pitch
