import torch
import torchaudio as ta
import os
import sys
import json
import re


def parse_vm(json_path: str, usernames: list, min_duration: float, max_duration: float, output_dir: str):
    assert os.path.exists(json_path)
    dirname = os.path.dirname(json_path)

    common_audio_length = dict.fromkeys(usernames, 0)

    with open(json_path, 'r') as file:
        data = json.load(file)
        for message in data["messages"]:
            if "from" in message:
                username = message["from"]
                if "media_type" in message:
                    media_type = message["media_type"]
                    if username in usernames and media_type == "voice_message":
                        duration = message["duration_seconds"]
                        if duration >= min_duration:
                            path_to_vm = os.path.join(dirname, message["file"])
                            voice_message, sr = ta.load(path_to_vm)
                            voice_message = voice_message[:int(max_duration * sr)]

                            duration = min(duration, max_duration)
                            common_audio_length[username] += duration

                            directory_save_file = os.path.join(output_dir, username)
                            os.makedirs(directory_save_file, exist_ok=True)

                            output_name = username + "_" + str(message["id"]) + ".wav"
                            print("Saving " + output_name )
                            output_save_path = os.path.join(directory_save_file, output_name)
                            ta.save(output_save_path, voice_message, sample_rate=sr)
    
    for user, length in common_audio_length.items():
        print(user + " - " + str(length))


def get_usernames(json_path: str, replace=True) -> list:
    assert os.path.exists(json_path)

    users = set()

    with open(json_path, 'r') as file:
        data = json.load(file)
        for message in data["messages"]:
            if "from" in message:
                user = message["from"]
                #if replace:
                #    user = re.sub(r'\s', '_', user)
                users.add(user)
    
    return users


def del_spaces(dirpath: str):
    for item in os.listdir(dirpath):
        path = os.path.join(dirpath, item)
        if os.path.isfile(path):
            item = str(item)
            new_name = '_'.join(item.split())
            dirname = os.path.dirname(path)
            new_path = os.path.join(dirname, new_name)
            os.rename(path, new_path)
        elif os.path.isdir(path):
            del_spaces(path)
            #item = str(item)
            new_name = '_'.join(item.split())
            dirname = os.path.dirname(path)
            new_path = os.path.join(dirname, new_name)
            os.rename(path, new_path)
                

if __name__ == "__main__":
    cfg = {
        "json_path" : "/home/comp/Рабочий стол/export_tg/2.json",
        "usernames" : ['Y G', 'Roman Romanovič', 'Gleb ㅤ'],
        "min_duration" : 2.0,
        "max_duration" : 25.0,
        "output_dir" : "/home/comp/Рабочий стол/vm"
    }
    #parse_vm(**cfg)
    #print(get_usernames("/home/comp/Рабочий стол/export_tg/2.json"))
    del_spaces("/home/comp/Рабочий стол/vm2")
    #s = 'Gleb ㅤ_50830.wav'
    #new_s = re.sub(r'\W+', '_', s)
    #print(new_s)