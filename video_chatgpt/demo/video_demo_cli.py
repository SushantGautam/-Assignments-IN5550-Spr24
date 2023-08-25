import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer
"""

docker run -it  --network host -v /home/sushant/D1/:/home/sushant/D1/  --gpus=0  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864   nvcr.io/nvidia/pytorch:23.07-py3
docker run -it  --network host  -v /home/sushant/D1/:/home/sushant/D1/  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined -u root  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install git-lfs


cd /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/
pip install -r requirements.txt
export PYTHONPATH="./:$PYTHONPATH"


python video_chatgpt/demo/video_demo_cli.py  --model-name ./LLaVA-7B-Lightening-v1-1/  --projection_path  ./video_chatgpt-7B.bin

"""

map_event_type = {
    "Ball out of play": "out",
    "Goal": "goal",
    "Shots": "shot",
    "Throw-in": "throw",
    "Foul": "foul",
}

# map_event_type = {
#     "Goal": "Goal",
#     "Shots": "Shots",
#     "Foul": "Foul",
#     "Ball out of play": "Ball out of play",
#     "Throw-in": "Throw-in",
# }

events_list = "', '".join(set(map_event_type.keys()))
events_list = f"'{events_list}'"

events_list_formatted = ", ".join(set(map_event_type.values()))
events_list_formatted = f"{events_list_formatted}"


questions = [
"Deeply analyze given soccer video clip which shows a particular game event type. "
# "Which one of the events from <"+events_list_formatted+"> best matches the event shown in the given video?"
"No pre-amble and output only in JSON with single key 'pred_evnt' with one of the events from <"+events_list_formatted+"> that best matches the event shown in given video. JSON output format: { ''pred_evnt'': ''...'' } ",   
]

print(questions[0])


from enum import Enum

class Status(Enum):
    SUCCESS = 1
    WRONG_ANSWER = 2
    KEY_NOT_FOUND = 3
    FAIL_PARSING = 4

import dirtyjson

def getKeyFromState(answer, key, value):
    last_response = ("{"+ answer + "}").replace("\n", '').replace("{{", '{').replace("}}", '}').replace("\\","")
    try:
        parsed_json = dict(dirtyjson.loads(last_response))
        if key in parsed_json:
            if parsed_json[key] == value:
                return (Status.SUCCESS, parsed_json)
            else:
                return (Status.WRONG_ANSWER, parsed_json, {"true_answer": value})
        else:
            return (Status.KEY_NOT_FOUND, parsed_json, {"expected_key": key, "true_answer": value})
    except:
        return (Status.FAIL_PARSING, last_response, {"expected_key": key, "true_answer": value})
# model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model("./LLaVA-7B-Lightening-v1-1/", "./video_chatgpt-7B.bin")
model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model("./LLaVA-7B-Lightening-v1-1/",  "/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/SoccerVideo-ChatGPT_7B-1.1_Checkpoints/mm_projector/checkpoint-48000.bin")

import glob
import random
random.seed(42)

vid_path = "/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/VideoMAE/SN10s-min5c-50/train"
mp4_files = glob.glob(f"{vid_path}/**/*.mp4")
random.shuffle(mp4_files)

from video_chatgpt.demo.chat import Chat

all_outputs = []
for vid_full_path in mp4_files:
    fileName = vid_full_path.split("/")[-1].replace(".mp4", "")
    _, ifReplay,CameraType, EventType = fileName.split("|")
    # all_events.append([vid_full_path, ifReplay, CameraType, EventType])
    # print("\nFile: ", [vid_full_path.split("/")[-1]])
    video_frames = load_video(vid_full_path)
    output_answer = video_chatgpt_infer(video_frames, questions[0], 'video-chatgpt_v1', model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
    true_event = map_event_type.get(EventType, "not_any")
    output = getKeyFromState(output_answer, "pred_evnt", true_event)
    print(output)
    all_outputs.append(str(output))


print("Output: \n", "\n".join([str(entry) for entry in all_outputs]))