import os
import argparse
import datetime
import json
import time
from video_chatgpt.video_conversation import default_conversation
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt_v1")
    parser.add_argument("--projection_path", type=str, required=False, default="")

    args = parser.parse_args()

    return args

args = parse_args()

# Initialize model, tokenizer, etc.
model, vision_tower, tokenizer, image_processor, video_token_len = \
    initialize_model(args.model_name, args.projection_path)
    
replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN


# Load video
# video_path = "video_chatgpt/demo/demo_sample_soccer/ball_out_of_play.mp4"
# video_path = "video_chatgpt/demo/demo_sample_soccer/foul.mp4"
# video_path = "video_chatgpt/demo/demo_sample_soccer/goal.mp4"
# video_path = "video_chatgpt/demo/demo_sample_soccer/injury.mp4"
# video_path = "video_chatgpt/demo/demo_sample_soccer/throw_in.mp4"
# video_path = "/home/sushant/SN10s-min5c-50/train/Goal/england_epl__2016-2017__2016-12-27-20-15Liverpool4-1StokeCity__1_224p_2060.55_2070.55|1|Close-UpPlayerOrFieldReferee|Goal.mp4"

def upload_video(video_path):
    _img_list = []
    # Upload the video to the chatbot
    llm_message = chat.upload_video(video_path, _img_list)
    return _img_list

def add_text(state, text, image, first_run):
    text = text[:1536]  # Hard cut-off
    # print("\ntext: ", text)
    # print("\nstate: ", state)
    if first_run:
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, image)
        state = default_conversation.copy()
    # print("\ntext1: ", text)
    # print("\nstate1: ", state)
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    # print(f"{first_run}, len: {len(text)}, state: {state.to_gradio_chatbot()}")
    # print("\nstate2: ", state)
    # print("first_run: ", first_run)
    return state



# Interactive chat loop
print("Chatbot initialized. Start chatting (type 'exit' to quit):")
# user_input = input("You: ")


# user_input ='Output in JSON with single key: { "event_type": <goal, foul, ball_throw_in, injury, ball_out_of_play or not_any> }. Your task is to analyze given soccer video clip and accurately determine event type from the optionsd.'
# while user_input.lower() != 'exit':
#     state = add_text(state, user_input, img_list, first_run)
#     *_, results = chat.answer(state, img_list, max_new_tokens=512, first_run=first_run, temperature=0.2)
#     state, _, img_list, first_run, _, _, _, _, _ = results
#     # os.system('cls' if os.name == 'nt' else 'clear')
#     print("\n\n".join([f"User: {entry[0]}\nBot: {entry[1]}" for entry in state.to_gradio_chatbot()]))
#     user_input = input("You: ")

# print("Chatbot session ended.")

questions = [
"Deeply analyze the given short soccer game video clip which shows a particular football game event. "
"No pre-amble and always output answer in JSON with single key 'pred_evnt' and value with the most representative event shown in the video with format: { 'pred_evnt': < only one of 'goal', 'ball_shot_no_goal', 'ball_out_of_boundary', 'ball_throw_in_from_sideline', 'foul_or_injury', 'not_any'> }. ",   
]



from enum import Enum

class Status(Enum):
    SUCCESS = 1
    WRONG_ANSWER = 2
    KEY_NOT_FOUND = 3
    FAIL_PARSING = 4

import dirtyjson

def getKeyFromState(state, key, value):
    last_response = state.to_gradio_chatbot()[-1][1].replace("\\", "").replace("\n", "").replace("'", '"') +"}"
    try:
        parsed_json = dict(dirtyjson.loads(last_response))
        if key in parsed_json:
            if parsed_json[key] == value:
                return (Status.SUCCESS, parsed_json)
            else:
                return (Status.WRONG_ANSWER, parsed_json, {"true_answer": value})
        else:
            return (Status.KEY_NOT_FOUND, parsed_json, {"expected_key": key})
    except:
        return (Status.FAIL_PARSING, last_response)


def AskGpt(question, video_path_):
    user_input = question
    first_run = True
    img_list = upload_video(video_path_)
    first_run= True
    state= None
    state = add_text(state, user_input, img_list, first_run)
    *_, results = chat.answer(state, img_list, max_new_tokens=512, first_run=first_run, temperature=0.2)
    state, _, img_list, first_run, _, _, _, _, _ = results
    # print("\n\n".join([f"User: {entry[0]}\nBot: {entry[1]}" for entry in state.to_gradio_chatbot()]))
    return state
# ['Ball out of play', 'Goal', 'Shots', 'Throw-in', 'Foul']

map_event_type = {
    "Ball out of play": "ball_out_of_play",
    "Goal": "goal",
    "Shots": "ball_shot_no_goal",
    "Throw-in": "ball_out_of_boundary",
    "Foul": "foul_or_injury",
}
    # only one of 'goal', 'ball_shot_no_goal', 'ball_out_of_play', 'ball_throw_in_from_sideline', 'foul', 'injury', '


import glob
import random

vid_path = "/home/sushant/SN10s-min5c-50/train"
mp4_files = glob.glob(f"{vid_path}/**/*.mp4")
random.shuffle(mp4_files)
mp4_files = mp4_files[:1]
from video_chatgpt.demo.chat import Chat

all_outputs = []
for vid_full_path in mp4_files:
    global chat
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token) #reset chat
    fileName = vid_full_path.split("/")[-1].replace(".mp4", "")
    _, ifReplay,CameraType, EventType = fileName.split("|")
    # all_events.append([vid_full_path, ifReplay, CameraType, EventType])
    print("\nFile: ", [vid_full_path.split("/")[-1]])
    state_ = AskGpt(questions[0], vid_full_path)
    true_event = map_event_type.get(EventType, "not_any")
    output = getKeyFromState(state_, "pred_evnt", true_event)
    print(output)
    all_outputs.append(output)

print("all_outputs: ", all_outputs)
# all_event_types = list(set([event[3] for event in all_events]))
