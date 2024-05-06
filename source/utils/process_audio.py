import torch
import torchaudio as ta
import torch.nn.functional as F
import os
from typing import Tuple
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm


def load_n_process_audio(filepath, output_dir, sr) -> Tuple[torch.Tensor, str]:
    assert os.path.exists(filepath)

    audio, fs = ta.load(filepath)
    filename = filepath.split(".")[0].split("/")[-1]
    changed = False

    # resample
    if fs != sr:
        print("Resampling")
        audio = ta.functional.resample(audio, fs, sr)
        filename += "_resampled"
        changed = True
    
    # make audio single channel
    if audio.shape[0] > 1:
        print("Treat as monochannel")
        audio = torch.mean(audio, dim=0, keepdim=True)
        filename += "_mono"
        changed = True

    # save changed audio
    if changed:
        directory_save_file = os.path.join(output_dir, filename)
        os.makedirs(directory_save_file, exist_ok=True)

        filename = filename + ".wav"
        filepath = os.path.join(directory_save_file, filename)
        ta.save(filepath, audio, sr)
    
    return [audio, filepath]


def cut_n_save(filepath, output_dir, csv_filepath):
    assert os.path.exists(filepath)
    assert os.path.exists(csv_filepath)

    audio, sr = ta.load(filepath)

    filename = filepath.split(".")[0].split("/")[-1]
    csv_filename = csv_filepath.split(".")[0].split("/")[-1]

    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')
    directory_save_file = os.path.join(output_dir, filename)
    directory_save_file_segments = os.path.join(directory_save_file, "segments")
    os.makedirs(directory_save_file_segments, exist_ok=True)

    for i, row in tqdm(df.iterrows()):
        start = row["start"]
        end = row["end"]
        id = row["id"]

        start = max(int(start * sr), 0)
        end = min(int(end * sr), audio.shape[-1])
        audio_segment = audio[..., start:end]

        save_segment_name = filename + '_' + str(id) + ".wav"
        save_segment_path = os.path.join(directory_save_file_segments, save_segment_name)

        ta.save(save_segment_path, audio_segment, sample_rate=sr)
        df.at[i, "path"] = save_segment_path

    new_csv_path = os.path.join(directory_save_file, (csv_filename + "_wpaths.csv"))
    df.to_csv(new_csv_path, sep=';', index=False, encoding='utf-8')
    return filepath, new_csv_path


def separate_audio_n_video(filepath, output_dir):
    assert os.path.exists(filepath)

    video_ext = filepath.split(".")[-1]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    video = VideoFileClip(filepath)
    audio = video.audio

    audio_save_path = os.path.join(directory_save_file, (filename + "_audio.wav"))
    video_save_path = os.path.join(directory_save_file, (filename + "_video." + video_ext))

    audio.write_audiofile(audio_save_path)
    video.write_videofile(video_save_path, audio=False)

    return [audio_save_path, video_save_path]


def align_audio_length(csv_filepath, output_dir, filename):
    assert os.path.exists(csv_filepath)

    csv_filename = csv_filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    directory_save_file_segments = os.path.join(directory_save_file, "segments")
    os.makedirs(directory_save_file_segments, exist_ok=True)

    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')

    for i, row in tqdm(df.iterrows()):
        start = row["start"]
        end = row["end"]
        audio_path = row["tts_path"]
        target_len = end - start

        audio = AudioSegment.from_wav(audio_path)

        len = audio.duration_seconds
        speedup = len / target_len
        audio = audio.speedup(playback_speed=speedup)

        segment_name = audio_path.split(".")[0].split("/")[-1]
        save_segment_name = segment_name + "_aligned.wav"
        save_segment_path = os.path.join(directory_save_file_segments, save_segment_name)

        audio.export(save_segment_path, format="wav")
        df.at[i, "aligned_tts"] = save_segment_path

    new_csv_path = os.path.join(directory_save_file, (csv_filename + "_aligned.csv"))
    df.to_csv(new_csv_path, sep=';', index=False, encoding='utf-8')
    return new_csv_path


def concat_segments(speech_path, background_path, csv_filepath, filename,
                    output_dir, join_video=False, video_path=""):
    assert os.path.exists(speech_path)
    assert os.path.exists(background_path)
    assert os.path.exists(csv_filepath)
    if join_video:
        assert os.path.exists(video_path)

    speech, sr_speech = ta.load(speech_path)
    background, sr_background = ta.load(background_path)

    assert (sr_speech == sr_background)

    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')

    for _, row in tqdm(df.iterrows()):
        start = row["start"]
        end = row["end"]
        segment_path = row["aligned_tts"]

        segment, sr_segment = ta.load(segment_path)
        assert (sr_speech == sr_segment)

        start = max(int(start * sr_segment), 0)
        end = min(int(end * sr_segment), speech.shape[-1])
        assert (end > start)

        target_len = end - start
        segment_len = segment.shape[-1]

        #len fuckups due to conversion from sec to samples on the previous steps
        if target_len <= segment_len:
            segment = segment[:, :target_len]
        else:
            padding_size = target_len - segment_len
            segment = F.pad(segment, pad=(0, padding_size), mode='constant', value=0)

        speech[:, start:end] = segment

    final_audio = background + speech
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    final_audio_name = filename + "_final_audio.wav"
    final_audio_save_path = os.path.join(directory_save_file, final_audio_name)
    ta.save(final_audio_save_path, final_audio, sample_rate=sr_speech)

    final_video_save_path = ""
    if join_video:
        video_ext = video_path.split(".")[-1]
        final_video_name = filename + "_final_video." + video_ext
        final_video_save_path = os.path.join(directory_save_file, final_video_name)

        video = VideoFileClip(video_path)
        audio = AudioFileClip(final_audio_save_path)

        video = video.set_audio(audio)
        video.write_videofile(final_video_save_path)

    return [final_audio_save_path, final_video_save_path]


if __name__ == "__main__":
    speech_path = "/home/comp/Рабочий стол/AutoDub/output/bsrnn/1_mono/1_mono_speech.wav"
    background_path = "/home/comp/Рабочий стол/AutoDub/output/bsrnn/1_mono/1_mono_background.wav"
    csv_filepath = "/home/comp/Рабочий стол/AutoDub/output/aligned_audio/1_mono_speech_resampled/1_mono_speech_resampled_asr_g_tr_wpaths_tts_wpaths.csv"
    filename = "1_mono_speech_resampled"
    output_dir = "/home/comp/Рабочий стол/AutoDub/output/final"
    join_video = False
    video_path = ""

    concat_segments(speech_path, background_path, csv_filepath, filename,
                    output_dir, join_video=False, video_path="")