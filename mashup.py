import os
import sys
import hydra
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import CONFIGS_PATH, resolve_paths
from source.inference.cascaded.inference_cascaded import inference_cascaded
from source.inference.vits.inference_vits import inference_vits
from source.utils.process_audio import concat_tracks

#FILEPATH = "$ROOT/input/test.mp4"
FILEPATH = ""

@hydra.main(config_path=str(CONFIGS_PATH), config_name="mashup")
def mashup(cfg):
    # if FILEPATH != "":
    #     cfg["filepath"] = FILEPATH
    # elif cfg["filepath"] is None:
    #     raise KeyError
    # else:
    #     pass
    cfg = resolve_paths(cfg, os.environ['ROOT'])
    # assert os.path.exists(cfg["filepath"])

    # filename = cfg["filepath"].split(".")[0].split("/")[-1]

    # print("Separating vocal")
    # cfg["cascaded"]["filepath"] = cfg["filepath"]
    # vocal_path, background_path = inference_cascaded(cfg["cascaded"])
    vocal_path = "/home/comp/Рабочий стол/Mashup/input/govnovoz_vocal.wav"
    filename = "govnovoz"
    background_path = "/home/comp/Рабочий стол/Mashup/input/govnovoz_background.wav"
    
    print("Converting vocal")
    cfg["vits"]["filepath"] = vocal_path
    converted_path = inference_vits(cfg["vits"])

    print("Concatenating")
    cfg["concatenate"]["vocal_path"] = converted_path
    cfg["concatenate"]["background_path"] = background_path
    cfg["concatenate"]["filename"] = filename
    concat_tracks(**cfg["concatenate"])


if __name__ == "__main__":
    os.environ['ROOT'] = os.getcwd()

    mashup()