import os
import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from omegaconf import OmegaConf
from source.inference.vits.preprocess import get_vec, get_ppg, get_pitch
from source.model.vits.synthesizer import SynthesizerInfer
from source.utils.feature_retrieval import DummyRetrieval, IRetrieval
from scipy.io.wavfile import write
from source.utils.pitch import load_csv_pitch



def svc_infer(model, retrieval: IRetrieval, spk, pit, ppg, vec, hp, device) -> torch.Tensor:
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
        #pitwav = model.source2wav(source)
        #write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mc = OmegaConf.load(cfg["model_config"])
    model = SynthesizerInfer(
        mc.data.filter_length // 2 + 1,
        mc.data.segment_size // mc.data.hop_length,
        mc)
    model.to(device)
    model.eval()

    checkpoint_dict = torch.load(cfg["checkpoint_path"], map_location="cpu")
    #saved_state_dict = checkpoint_dict["model_g"]
    saved_state_dict = checkpoint_dict["gen"]
    state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(saved_state_dict)
    retrieval = DummyRetrieval()
    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    vc_save_path = os.path.join(directory_save_file, (filename + "_converted.wav"))

    if cfg["ppg"] == "":
        cfg["process_ppg"]["filepath"] = filepath
        cfg["ppg"] = get_ppg(cfg["process_ppg"])

    if cfg["vec"] == "":
        cfg["process_vec"]["filepath"] = filepath
        cfg["vec"] = get_vec(cfg["process_vec"])

    if cfg["pitch"] == "":
        cfg["process_pitch"]["filepath"] = filepath
        cfg["pitch"] = get_pitch(cfg["process_pitch"])

    for elem in ["spk", "ppg", "vec", "pitch"]:
        assert os.path.exists(cfg[elem])

    spk = np.load(cfg["spk"])
    spk = torch.FloatTensor(spk)

    ppg = np.load(cfg["ppg"])
    ppg = np.repeat(ppg, 2, 0) # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    vec = np.load(cfg["vec"])
    vec = np.repeat(vec, 2, 0) # 320 PPG -> 160 * 2
    vec = torch.FloatTensor(vec)

    pitch = load_csv_pitch(cfg["pitch"])
    print("pitch shift: ", cfg["shift"])
    if (cfg["shift"] == 0):
        pass
    else:
        pitch = np.array(pitch)
        source = pitch[pitch > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = cfg["shift"]
        shift = 2 ** (shift / 12)
        pitch = pitch * shift
    pitch = torch.FloatTensor(pitch)

    converted = svc_infer(model, retrieval, spk, pitch, ppg, vec, mc, device)
    write(vc_save_path, mc.data.sampling_rate, converted)

    #ta.save(vc_save_path, converted, sample_rate=mc.data.sampling_rate)

    return vc_save_path


    

if __name__ == "__main__":
    cfg = {
        "checkpoint_path" : "/home/comp/Рабочий стол/Mashup/checkpoints/vits/checkpoint-epoch64.pth",
        "spk" : "/home/comp/Рабочий стол/Mashup/data_svc_pg/singer/petr_grinberg.spk.npy",
        "ppg" : "/home/comp/Рабочий стол/Mashup/output/whisper/govnovoz_vocal/govnovoz_vocal_ppg.npy",
        "vec" : "/home/comp/Рабочий стол/Mashup/output/hubert/govnovoz_vocal/govnovoz_vocal_hubert.npy",
        "pitch" : "/home/comp/Рабочий стол/Mashup/output/pitch/govnovoz_vocal/govnovoz_vocal_pitch.csv",
        "shift" : 0,
        "model_config" : "/home/comp/Рабочий стол/Mashup/source/configs/vits/base.yaml",
        "filepath" : "/home/comp/Рабочий стол/samples/govnovoz_vocal.wav",
        "output_dir" : "/home/comp/Рабочий стол/Mashup/output/vits"
    }
    inference_vits(cfg)

