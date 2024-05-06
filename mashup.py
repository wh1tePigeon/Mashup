import os
import sys
import hydra
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import CONFIGS_PATH, resolve_paths
from source.inference import inference_bsrnn, inference_vad, inference_asr, translate_file_google, lazy_tts
from source.utils.process_audio import cut_n_save, separate_audio_n_video, align_audio_length, concat_segments

#FILEPATH = "$ROOT/input/test.mp4"
FILEPATH = "$ROOT/input/1.wav"

@hydra.main(config_path=str(CONFIGS_PATH), config_name="dub")
def dub(cfg):
    if FILEPATH != "":
        cfg["filepath"] = FILEPATH
    elif cfg["filepath"] is None:
        raise KeyError
    else:
        pass
    cfg = resolve_paths(cfg, os.environ['ROOT'])
    assert os.path.exists(cfg["filepath"])

    audio_path = ""
    video_path = ""
    wvideo = False

    file_ext = cfg["filepath"].split(".")[-1]
    if file_ext == "mp4":
        print("Separating audio from video")
        cfg["separate_audio"]["filepath"] =  cfg["filepath"]
        audio_path, video_path = separate_audio_n_video(**cfg["separate_audio"])
        wvideo = True
    elif file_ext == "wav":
        audio_path = cfg["filepath"]

    print("Separating speech")
    cfg["bsrnn"]["filepath"] = audio_path
    speech_path, background_path = inference_bsrnn(cfg["bsrnn"])
    filename = speech_path.split(".")[0].split("/")[-1]
    
    print("Detecting voice activity")
    cfg["vad"]["filepath"] = speech_path
    filepath, vad_boundaries_path = inference_vad(cfg["vad"])

    print("Speech recognition")
    cfg["asr"]["filepath"] = filepath
    cfg["asr"]["boundaries"] = vad_boundaries_path
    transcribed_path = inference_asr(cfg["asr"])

    print("Translating")
    cfg["tr"]["filepath"] = transcribed_path
    translated_csv_path = translate_file_google(cfg["tr"])

    print("Cutting audio")
    cfg["cut"]["filepath"] = cfg["asr"]["filepath"]
    cfg["cut"]["csv_filepath"] = translated_csv_path
    _, cutted_csv_path = cut_n_save(**cfg["cut"])

    print("TTS")
    cfg["tts"]["csv_filepath"] = cutted_csv_path
    cfg["tts"]["filename"] = filename
    cfg["tts"]["target_sr"] = cfg["bsrnn"]["sr"]
    tts_csv_path = lazy_tts(**cfg["tts"])

    print("Align tts")
    cfg["align_audio"]["csv_filepath"] = tts_csv_path
    cfg["align_audio"]["filename"] = filename
    aligned_audio_csv_path = align_audio_length(**cfg["align_audio"])

    print("Concatenate")
    cfg["concatenate"]["speech_path"] = speech_path
    cfg["concatenate"]["background_path"] = background_path
    cfg["concatenate"]["csv_filepath"] = aligned_audio_csv_path
    cfg["concatenate"]["filename"] = filename
    cfg["concatenate"]["join_video"] = wvideo
    cfg["concatenate"]["video_path"] = video_path
    concat_segments(**cfg["concatenate"])


if __name__ == "__main__":
    os.environ['ROOT'] = os.getcwd()

    dub()