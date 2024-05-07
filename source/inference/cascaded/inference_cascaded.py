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
from source.model.cascaded.cascaded import CascadedNet
from source.model.cascaded.separator import Separator
from source.utils.spec_utils import wave_to_spectrogram, spectrogram_to_wave

def inference_cascaded(cfg):
    # device, device_ids = prepare_device(cfg["n_gpu"])
    # arch = OmegaConf.load(cfg["model"])
    # model = instantiate(arch)
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    # separator = 

    # checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    # state_dict = checkpoint["state_dict"]
    # model.load_state_dict(state_dict)
    # model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    model = CascadedNet(cfg["n_fft"], cfg["hop_length"], 32, 128)
    model.to(device)
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt)
    

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]

    if os.path.isfile(filepath):
        sr = 44100
        audio, _ = librosa.load(
        filepath, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        filename = filepath.split(".")[0].split("/")[-1]
        
        directory_save_file = os.path.join(output_dir, filename)
        os.makedirs(directory_save_file, exist_ok=True)
        
        vocal_save_path = os.path.join(directory_save_file, (filename + "_vocal.wav"))
        background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))
        
        if audio.ndim == 1:
        # mono to stereo
            audio = np.asarray([audio, audio])

        audio_spec = wave_to_spectrogram(audio, cfg["hop_length"], cfg["n_fft"])

        sp = Separator(
            model=model,
            device=device,
            batchsize=cfg["batchsize"],
            cropsize=cfg["cropsize"],
            postprocess=cfg["postprocess"]
        )

        background_spec, vocal_spec = sp.separate(audio_spec)

        background = torch.from_numpy(spectrogram_to_wave(background_spec, hop_length=cfg["hop_length"]))
        vocal = torch.from_numpy(spectrogram_to_wave(vocal_spec, hop_length=cfg["hop_length"]))

        ta.save(vocal_save_path, vocal, sample_rate=sr)
        ta.save(background_save_path, background, sample_rate=sr)



        # audio, filepath = load_n_process_audio(filepath, output_dir, sr)
        # audio = audio.reshape(1, 1, -1)
  
        # # move audio to gpu
        # audio = audio.to(device)

        # with torch.inference_mode():
        #     def forward(audio):
        #         _, output = model({"audio": {"mixture": audio},})
        #         return output["audio"]
            
        #     if audio.shape[-1] / sr > 12:
        #         fader = OverlapAddFader(window_type=cfg["window_type"],
        #                                 chunk_size_second=cfg["chunk_size_second"],
        #                                 hop_size_second=cfg["hop_size_second"],
        #                                 fs=sr,
        #                                 batch_size=cfg["batch_size"])
                
        #         output = fader(audio,
        #                         lambda a: forward(a))
                
        #     else:
        #         output = forward(audio)
            
        #     speech = output["audio"]["speech"]
        #     speech = speech.reshape(1, -1)
        #     audio = audio.reshape(1, -1)
        #     background = audio - speech

        #     filename = filepath.split(".")[0].split("/")[-1]

        #     directory_save_file = os.path.join(output_dir, filename)
        #     os.makedirs(directory_save_file, exist_ok=True)
            
        #     speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
        #     background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))

        #     speech = speech.to("cpu")
        #     background = background.to("cpu")

        #     ta.save(speech_save_path, speech, sample_rate=sr)
        #     ta.save(background_save_path, background, sample_rate=sr)

        #     return [speech_save_path, background_save_path]


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