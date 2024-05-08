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
from source.model.vits.synthesizer import SynthesizerInfer
from source.model.cascaded.separator import Separator
from source.utils.spec_utils import wave_to_spectrogram, spectrogram_to_wave
from source.utils.feature_retrieval import DummyRetrieval
from scipy.io.wavfile import write
from source.utils.pitch import load_csv_pitch
from source.utils.feature_retrieval import IRetrieval


def svc_infer(model, retrieval: IRetrieval, spk, pit, ppg, vec, hp, device):
    len_pit = pit.size()[0]
    len_vec = vec.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_vec)
    len_min = min(len_min, len_ppg)
    pit = pit[:len_min]
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]

    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        source = pit.unsqueeze(0).to(device)
        source = model.pitch2source(source)
        pitwav = model.source2wav(source)
        write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

        hop_size = hp.data.hop_length
        all_frame = len_min
        hop_frame = 10
        out_chunk = 2500  # 25 S
        out_index = 0
        out_audio = []

        while (out_index < all_frame):

            if (out_index == 0):  # start frame
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = out_index - hop_frame
                cut_s_out = hop_frame * hop_size

            if (out_index + out_chunk + hop_frame > all_frame):  # end frame
                cut_e = all_frame
                cut_e_out = -1
            else:
                cut_e = out_index + out_chunk + hop_frame
                cut_e_out = -1 * hop_frame * hop_size

            sub_ppg = retrieval.retriv_whisper(ppg[cut_s:cut_e, :])
            sub_vec = retrieval.retriv_hubert(vec[cut_s:cut_e, :])
            sub_ppg = sub_ppg.unsqueeze(0).to(device)
            sub_vec = sub_vec.unsqueeze(0).to(device)
            sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s *
                             hop_size:cut_e * hop_size].to(device)
            sub_out = model.inference(
                sub_ppg, sub_vec, sub_pit, spk, sub_len, sub_har)
            sub_out = sub_out[0, 0].data.cpu().detach().numpy()

            sub_out = sub_out[cut_s_out:cut_e_out]
            out_audio.extend(sub_out)
            out_index = out_index + out_chunk

        out_audio = np.asarray(out_audio)
    return out_audio


def inference_vits(cfg):
    # if (cfg["ppg"] == None):
    #     cfg["ppg"] = "svc_tmp.ppg.npy"
    #     print(
    #         f"Auto run : python whisper/inference.py -w {cfg['wave']} -p {cfg['ppg']}")
    #     os.system(f"python whisper/inference.py -w {cfg['wave']} -p {cfg['ppg']}")

    # if (cfg["vec"] == None):
    #     cfg["vec"] = "svc_tmp.vec.npy"
    #     print(
    #         f"Auto run : python hubert/inference.py -w {cfg['wave']} -v {cfg['vec']}")
    #     os.system(f"python source/inference/hubert/inference_hubert.py -w {cfg['wave']} -v {cfg['vec']}")

    # if (cfg["pit"] == None):
    #     cfg["pit"] = "svc_tmp.pit.csv"
    #     print(
    #         f"Auto run : python pitch/inference.py -w {cfg['wave']} -p {cfg['pit']}")
    #     os.system(f"python source/utils/pitch.py -w {cfg['wave']} -p {cfg['pit']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    hp = OmegaConf.load(cfg)
    model = SynthesizerInfer(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
    model.to(device)
    model.eval()

    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt)

    retrieval = DummyRetrieval()

    spk = np.load(cfg["spk"])
    spk = torch.FloatTensor(spk)

    ppg = np.load(cfg["ppg"])
    ppg = np.repeat(ppg, 2, 0) # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    vec = np.load(cfg["vec"])
    vec = np.repeat(vec, 2, 0) # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)

    pit = load_csv_pitch(cfg["pit"])
    print("pitch shift: ", cfg["shift"])
    if (cfg["shift"] == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = cfg["shift"]
        shift = 2 ** (shift / 12)
        pit = pit * shift
    pit = torch.FloatTensor(pit)

    out_audio = svc_infer(model, retrieval, spk, pit, ppg, vec, hp, device)
    write("svc_out.wav", hp.data.sampling_rate, out_audio)




    
    # ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    # model.load_state_dict(ckpt)
    

    # filepath = cfg["filepath"]
    # output_dir = cfg["output_dir"]

    # if os.path.isfile(filepath):
    #     sr = 44100
    #     audio, _ = librosa.load(
    #     filepath, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

    #     filename = filepath.split(".")[0].split("/")[-1]
        
    #     directory_save_file = os.path.join(output_dir, filename)
    #     os.makedirs(directory_save_file, exist_ok=True)
        
    #     vocal_save_path = os.path.join(directory_save_file, (filename + "_vocal.wav"))
    #     background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))
        
    #     if audio.ndim == 1:
    #     # mono to stereo
    #         audio = np.asarray([audio, audio])

    #     audio_spec = wave_to_spectrogram(audio, cfg["hop_length"], cfg["n_fft"])

    #     sp = Separator(
    #         model=model,
    #         device=device,
    #         batchsize=cfg["batchsize"],
    #         cropsize=cfg["cropsize"],
    #         postprocess=cfg["postprocess"]
    #     )

    #     background_spec, vocal_spec = sp.separate(audio_spec)

    #     background = torch.from_numpy(spectrogram_to_wave(background_spec, hop_length=cfg["hop_length"]))
    #     vocal = torch.from_numpy(spectrogram_to_wave(vocal_spec, hop_length=cfg["hop_length"]))

    #     ta.save(vocal_save_path, vocal, sample_rate=sr)
    #     ta.save(background_save_path, background, sample_rate=sr)

    #     return [vocal_save_path, background_save_path]


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
    print(torch.cuda.is_available())
    #inference_cascaded(cfg)