import os
import numpy as np
import soundfile as sf


def main():
    path_to_inst = "/home/comp/Рабочий стол/musdb18_changed/instruments"
    path_to_mix = "/home/comp/Рабочий стол/musdb18_changed/mixtures"
    from source.utils.musdb import DB
    mus = DB(root="/home/comp/Рабочий стол/musdb18")
    for track in mus:
        name = track.name
        print(name)
        save_path_inst =  os.path.join(path_to_inst, (name + "_inst.wav"))
        save_path_mix =  os.path.join(path_to_mix, (name + "_mix.wav"))

        mixture = track.audio
        vocals = track.targets['vocals'].audio
        inst = mixture - vocals

        sf.write(save_path_mix, mixture, samplerate=44100)
        sf.write(save_path_inst, inst, samplerate=44100)


if __name__ == "__main__":
    main()