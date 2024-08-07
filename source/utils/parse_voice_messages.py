"""
In the desktop version of Telegram, open the chat from which you want to download voice messages.
Click on the three dots in the upper right corner of the screen -> export chat history.
Select "voice messages" and specify the format "json".
In the configs, specify the path to the "voice_messages" folder and the "result.json" file
On a Linux Mint system, they are usually saved in
/home/<username>/.var/app/org.telegram.desktop/data/TelegramDesktop/tdata/temp_data/ChatExport_<date>
"""

import os
import json
import re
import torchaudio as ta

def remove_non_litters(s: str) -> str:
    new_s = re.sub("[^A-Za-z0-9.]", "_", s)
    new_s = re.sub(r'\s', '_', new_s)
    return new_s


def parse_vm(json_path: str, users_id: list, min_duration: float, max_duration: float, output_dir: str):
    assert os.path.exists(json_path)
    dirname = os.path.dirname(json_path)

    common_audio_length = dict.fromkeys(users_id, 0)

    with open(json_path, 'r') as file:
        data = json.load(file)
        for message in data["messages"]:
            if "from_id" in message:
                user_id = message["from_id"]
                if "media_type" in message:
                    media_type = message["media_type"]
                    if user_id in users and media_type == "voice_message":
                        duration = message["duration_seconds"]
                        if duration >= min_duration:
                            path_to_vm = os.path.join(dirname, message["file"])
                            voice_message, sr = ta.load(path_to_vm)
                            voice_message = voice_message[:int(max_duration * sr)]

                            duration = min(duration, max_duration)
                            common_audio_length[user_id] += duration

                            directory_save_file = os.path.join(output_dir, user_id)
                            os.makedirs(directory_save_file, exist_ok=True)

                            output_name = user_id + "_" + str(message["id"]) + ".wav"
                            print("Saving " + output_name )
                            output_save_path = os.path.join(directory_save_file, output_name)
                            ta.save(output_save_path, voice_message, sample_rate=sr)

    return common_audio_length


def get_users(json_path: str, replace=True) -> list:
    assert os.path.exists(json_path)

    users = {}

    with open(json_path, 'r') as file:
        data = json.load(file)
        for message in data["messages"]:
            if "from_id" in message:
                if message["from_id"] not in users:
                    users[message["from_id"]] = message["from"]
    
    return users


def remove_non_litters_from_dir(dirpath: str):
    for item in os.listdir(dirpath):
        path = os.path.join(dirpath, item)
        if os.path.isfile(path):
            new_name = remove_non_litters(item)
            dirname = os.path.dirname(path)
            new_path = os.path.join(dirname, new_name)
            os.rename(path, new_path)
        elif os.path.isdir(path):
            remove_non_litters_from_dir(path)
            new_name = remove_non_litters(item)
            dirname = os.path.dirname(path)
            new_path = os.path.join(dirname, new_name)
            os.rename(path, new_path)

        
def print_common_audio_length(common_audio_length, users):
    for id, name in users:
        print(str(name) + " - " + str(common_audio_length[id]))
                

if __name__ == "__main__":
    cfg = {
        "json_path" : "/home/comp/Рабочий стол/export_tg/2.json",
        "users_id" : [],
        "min_duration" : 2.0,
        "max_duration" : 25.0,
        "output_dir" : "/home/comp/Рабочий стол/vm3"
    }
    users = get_users(cfg["json_path"])
    cfg["users_id"] = users.keys()
    parse_vm(**cfg)
