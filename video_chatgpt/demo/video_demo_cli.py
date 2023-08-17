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


# Create chat for the demo
from video_chatgpt.demo.chat import Chat
chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)

# Load video
video_path = "video_chatgpt/demo/demo_sample_soccer/ball_out_of_play.mp4"

def upload_video(video_path):
    _img_list = []
    # Upload the video to the chatbot
    llm_message = chat.upload_video(video_path, _img_list)
    return _img_list

def add_text(state, text, image, first_run):
    text = text[:1536]  # Hard cut-off
    if first_run:
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, image)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    print(f"{first_run}, len: {len(text)}, state: {state.to_gradio_chatbot()}")
    return state


img_list = upload_video(video_path)
first_run= True
state= None

# Interactive chat loop
print("Chatbot initialized. Start chatting (type 'exit' to quit):")
# user_input = input("You: ")

user_input ="what is the score?"

while user_input.lower() != 'exit':
    state = add_text(state, user_input, img_list, first_run)
    *_, results = chat.answer(state, img_list, max_new_tokens=512, first_run=first_run, temperature=0.2)
    state, _, img_list, first_run, _, _, _, _, _ = results
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n\n".join([f"User: {entry[0]}\nBot: {entry[1]}" for entry in state.to_gradio_chatbot()]))
    user_input = input("You: ")

print("Chatbot session ended.")
