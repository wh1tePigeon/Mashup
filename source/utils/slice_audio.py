import os
import sys
import torch
import torchaudio as ta
from speechbrain.inference.VAD import VAD
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.utils.process_audio import load_n_process_audio


SR = 16000


def process_small_segments(ouput_dir: str, min_duration: float, segments: list, ind: int):
    os.makedirs(ouput_dir, exist_ok=True)

    tmp = torch.empty(1, 0)
    for segment in segments:
        tmp = torch.cat((tmp, segment), dim=-1)
        tmp_len_seconds = tmp.shape[-1] / SR
        if tmp_len_seconds >= min_duration:
            output_filename = str(ind) + ".wav"
            print("Saving " + output_filename)
            tmp_save_path = os.path.join(ouput_dir, output_filename)
            ta.save(tmp_save_path, tmp, SR)
            ind += 1
            tmp = tmp = torch.empty(1, 0)


def slice(dirpath: str, ouput_dir: str, min_duration: float, max_duration: float, vad_cfg: dict):
    assert os.path.exists(dirpath)
    os.makedirs(ouput_dir, exist_ok=True)

    total_length_sec = 0
    ind = 0
    segments_less_than_min = []

    for filename in os.listdir(dirpath):
        if filename.endswith(".wav"):
            filepath = os.path.join(dirpath, filename)
            audio, sr = ta.load(filepath)
            if sr != SR:
                sr=SR
                new_dirpath = os.path.join(os.path.dirname(dirpath), os.path.dirname(dirpath) + "_resampled")
                audio, filepath = load_n_process_audio(filepath, new_dirpath, SR, create_dir=False)


            vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                            savedir=vad_cfg["checkpoint_path"])
            
            boundaries = vad.get_speech_segments(audio_file=filepath, apply_energy_VAD=False)
            #                                     apply_energy_VAD=True,
            #                                     activation_th=cfg["activation_th"],
            #                                     deactivation_th=cfg["deactivation_th"],
            #                                     close_th=cfg["close_th"],
            #                                     len_th=cfg["len_th"])

            # somehow this works better ¯\_(ツ)_/¯
            boundaries = vad.energy_VAD(filepath, boundaries,
                                        activation_th=vad_cfg["activation_th"],
                                        deactivation_th=vad_cfg["deactivation_th"])
            boundaries = vad.merge_close_segments(boundaries, close_th=vad_cfg["close_th"])
            boundaries = vad.remove_short_segments(boundaries, len_th=vad_cfg["len_th"])
            
            
            for [start_time, end_time] in boundaries:
                start = int(start_time * SR)
                end = int(end_time * SR)
                total_length_sec += (end_time - start_time)
                segment = audio[..., start:end]
                segment_len = segment.shape[-1]
                segment_len_seconds = segment.shape[-1] / SR

                if min_duration <= segment_len_seconds <= max_duration:
                    output_filename = str(ind) + ".wav"
                    print("Saving " + output_filename)
                    segment_save_path = os.path.join(ouput_dir, output_filename)
                    ta.save(segment_save_path, segment, SR)
                    ind += 1

                elif min_duration > segment_len_seconds:
                    segments_less_than_min.append(segment)

                elif segment_len_seconds > max_duration:
                    max_duration_len = max_duration * SR
                    amount_of_segments = segment_len // max_duration_len

                    for i in range(0, amount_of_segments - 1):
                        sub_segment = segment[i * max_duration_len : (i + 1) * max_duration_len]
                        output_filename = str(ind) + ".wav"
                        print("Saving " + output_filename)
                        sub_segment_save_path = os.path.join(ouput_dir, output_filename)
                        ta.save(sub_segment_save_path, sub_segment, SR)
                        ind += 1
                    res = amount_of_segments * max_duration_len

                    if res != segment_len:
                        residual_segment = segment[res :]

                        if min_duration <= residual_segment.shape[-1] / SR:
                            output_filename = str(ind) + ".wav"
                            print("Saving " + output_filename)
                            residual_segment_save_path = os.path.join(ouput_dir, output_filename)
                            ta.save(residual_segment_save_path, residual_segment, SR)
                            ind += 1
                        else:
                            segments_less_than_min.append(residual_segment)
    return total_length_sec, ind, segments_less_than_min


if __name__ == "__main__":
    cfg = {
        "dirpath" : "/home/comp/Рабочий стол/data",
        "ouput_dir" : "/home/comp/Рабочий стол/pg_output",
        "min_duration" : 4.0,
        "max_duration" : 20.0,
        "vad_cfg" :  {
        "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/vad",
        "activation_th" : 0.8,
        "deactivation_th" : 0.0,
        "close_th" : 0.250,
        "len_th" : 0.250
        }
    }
    total_length_sec, ind, segments_less_than_min = slice(**cfg)
    print(total_length_sec)
    process_small_segments(cfg["ouput_dir"], 2.0, segments_less_than_min, ind)
