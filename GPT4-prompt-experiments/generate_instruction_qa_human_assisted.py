# Required Libraries
import argparse
import ast
import json
import os
import time
import warnings
from datetime import datetime
import pandas as pd
import openai
from tqdm import tqdm
from multiprocessing.pool import Pool


# Suppressing all warnings
warnings.filterwarnings("ignore")

import tiktoken

model_name = "gpt-3.5-turbo"
encoder = tiktoken.encoding_for_model(model_name)


def save_analytics(caption, conversation, reply, response_dict):
    encoded_text = encoder.encode(
        "\n".join([message["content"] for message in conversation])
    )
    token_count = len(encoded_text)
    encoded_text = encoder.encode(reply)
    token_count_reply = len(encoded_text)
    print(token_count_reply, "tokens outputted.")
    print("Assistant: gpt_response_json=", reply)
    print("total tokens:", token_count + token_count_reply)
    # append to csv file, time only, input, output, inout tokens, output tokens
    # file name is date today
    log_file = f"analytics/gpt4-analytics{ datetime.now().strftime('%Y-%m-%d') }.csv"
    with open(log_file, "a") as f:
        caption_ = caption.replace("\n", "\\n")
        f.write(
            f"{datetime.now().strftime('%H:%M:%S')},{caption_},{response_dict},{token_count},{token_count_reply},{token_count + token_count_reply}\n"
        )


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Descriptive question-answer-generation-using-GPT-3"
    )
    parser.add_argument("--input_type", required=True, help="e1 or e2")
    parser.add_argument(
        "--num_tasks", required=False, type=int, help="Number of splits.", default=1
    )
    parser.add_argument(
        "--creative",
        required=False,
        type=bool,
        help="Generate Creative Based QA pairs.",
        default=False,
    )
    parser.add_argument(
        "--summary",
        required=False,
        type=bool,
        help="Generate Summary Based QA pairs.",
        default=False,
    )
    parser.add_argument(
        "--caption",
        required=False,
        type=bool,
        help="Generate Caption Based QA pairs.",
        default=False,
    )

    return parser.parse_args()

import requests_cache
e1_data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=353977511&single=true&output=csv", index_col=0,)
e2_data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=957600063&single=true&output=csv", index_col=0)



def annotate(gt_file, caption_files, output_dir, args):
    """
    Generates question and answer pairs based on video captions using OpenAI GPT-3.
    """

    for file in tqdm(caption_files):
        key = file.split(".")[0].split("_")[0]
        caption = gt_file[key]

        # breakpoint()
        # find event type shown in  the video
        if str(args.input_type) == "e1":
            event_name = e1_data.loc[int(file.split('_')[0]), "label"]
            event_info ="The only visible event is: "+ event_name + ". "
        elif str(args.input_type) == "e2":
            events_name = e2_data.loc[int(file.split('_')[0]), "Pair-label"].split("->")
            event_info ="Only two events are shown: "+ events_name[0] + " and then " + events_name[1] + ". "
        else:
            print("Error: Invalid input type. Exiting...", key)
            exit()
        

        print(caption, "\n\n")

        # tmp #########

        # if file.split(".")[0].split("_")[1] == 'summary':
        #     response_dict = {'summarrt_qa': 'summary_qa'}
        # elif file.split(".")[0].split("_")[1] == 'caption':
        #     response_dict = {'caption_qa': 'caption_qa'}
        # elif file.split(".")[0].split("_")[1] == 'creative':
        #     response_dict = {'creative_qa': 'creative_qa'}

        # end tmp #####

        # Generate QA pairs with OpenAI GPT-3: Summarization
        if file.split(".")[0].split("_")[1] == "summary":
            message = [
                {
                    "role": "system",
                    "content": "You play two roles: a human asking questions related to summarizing a short soccer game clip and an intelligent chatbot designed for video summarization and dense captioning. "
                    "Your task is video summarization. "
                    "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                    "Your task is to play the role of a human who asks three questions related to summarizing the video and then play the role of an AI assistant that provides paraphrased answers based on the video content and the provided caption."
                    "------"
                    "##TASK:"
                    "Users will provide a caption of a video, and you will generate a set of three conversation-like questions related to summarizing the video. "+ event_info+ 
                    "The caption can mention major events not shown in the clip. The questions and answers can be very similar, but they should all focus on summarizing the key event event shown. "
                    "Each answers should be distinct paraphrased versions of the provided caption about the key visible events. Don't talk about viewers and fans."
                    "You have information about the video based on the provided caption and have to summarize the visible game events in it. "
                    "Generate THREE different diverse-type questions asking to summarize the video and provide detailed answers to each based on the caption. "
                    "------"
                    "##INSTRUCTIONS:"
                    "- The questions must be like a human conversation and focused on summarizing the video. "
                    "- The answers must be paraphrased versions of the provided caption, and they should be detailed and descriptive. "
                    "- Refrain from mentioning the actual names of the players and teams in the answer."
                    "------"
                    "##SAMPLE QUESTIONS:"
                    "- Can you provide a summary of the game video?"
                    "- What are the main events shown in the video?"
                    "- What's the essence of the game's dynamics?"
                    "- Could you briefly describe the video content?",
                },
                {
                    "role": "user",
                    "content": f"The video caption is: {caption}."
                    "Please generate the response in the form of a Python JSON, where JSON strings starting with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                    "Emphasize that the questions and answers can be very similar, but they should all focus on summarizing the video content."
                    "The response should look EXACTLY like this : {'Q1': 'Your first question here...', 'A1': 'Your first answer here...', 'Q2': 'Your second question here...', 'A2': 'Your second answer here...', 'Q3': 'Your third question here...', 'A3': 'Your third answer here...'}. ",

                },
            ]
            completion_0 = openai.ChatCompletion.create(
                messages=message,
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
            )
            # Extract Summary Based QA pairs
            # Convert response to a list of dictionary.
            response_message = completion_0["choices"][0]["message"]["content"]
            # response_message ='{"s": "test"}' #REM

        elif file.split(".")[0].split("_")[1] == "caption":
            # Generate QA pairs with OpenAI GPT-3: Caption Based
            # Answers specifically restricted to information in the caption
            message = [
                    {
                        "role": "system",
                        "content": "You play two roles: a human asking questions related to a short soccer video clip and an intelligent chatbot designed to help people understand specific events within the clip. "
                        "Your task is to focus on soccer video summarization, which will be utilized by users to comprehend key moments in soccer matches through various questions based on the video content. "
                        "This summarization will assist in applications like analyzing game highlights, generating summaries for sports content platforms, creating brief overviews for coaching analysis, or providing quick updates for fans. "
                        "You will first act as a human inquiring about specific events in a soccer match and then switch roles to an AI assistant providing detailed information based on the video's content."
                        "------"
                        "##TASK:"
                        "You will be given a caption of a specific events from a short soccer video clip. Based on this caption, you will generate a set of conversational-style questions and answers related to the visible events. "+ event_info+ 
                        "The questions should be crafted to extract information DIRECTLY from the provided caption, so that it or parts of it can serve as the answers. "
                        "Generate THREE different descriptive and conversational style questions and detailed answers based on the given information."
                        "------"
                        "##INSTRUCTIONS:"
                        "- The questions must be conversational and directly related to the events in the soccer video clip. "
                        "- The questions should be designed to extract information DIRECTLY from the given caption, so that it or parts of it can serve as the answers. "
                        "- The answers must be detailed, descriptive, and should directly reference the information provided. "
                        "- The questions can focus on player actions, game strategies, scoring opportunities, defensive tactics, or any key moments in the clip. "
                        "------"
                        "##SAMPLE QUESTIONS (based on given caption and event type):"
                        "- How did the player score the goal in the clip?"
                        "- What defensive strategy did the team use to prevent the goal?"
                        "- Describe the sequence of passes that led to the goal."
                        "- Was there an offside violation in the buildup to the goal?"
                        "- How did the goalkeeper react to the shot?"
                    },
                    {
                        "role": "user",
                        "content": f"The video caption is: {caption}. "
                        "Please generate the response in the form of a Python JSON, where JSON strings starting with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                        "The response should look EXACTLY like this : {'Q1': 'Your first question here...', 'A1': 'Your first answer here...', 'Q2': 'Your second question here...', 'A2': 'Your second answer here...', 'Q3': 'Your third question here...', 'A3': 'Your third answer here...'}. "
                        "Emphasize that ALL THREE questions must be designed to extract information DIRECTLY from the given caption, so that it or parts of it can serve as the answers, and provide detailed and descriptive answers."
                    }
            ]


            completion_1 = openai.ChatCompletion.create(
                messages=message,
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
            )

            # Extract Caption Based QA pairs
            # Convert response to a list of dictionary.
            response_message = completion_1["choices"][0]["message"]["content"]

        elif file.split(".")[0].split("_")[1] == "creative":
            # Generate QA pairs with OpenAI GPT-3: Creative Based
            # TODO: Limit to samples with lengthy GT captions
            message = [
                {
                    "role": "system",
                    "content": "You play two roles: a human asking questions related to a video and an intelligent chatbot designed to help people find information from a given short soccer game clip. "
                    "You play two roles: a human asking creative questions related to a video and an intelligent chatbot designed to help people explore imaginative aspects of a given video. "
                    "Your task is to generate a conversation that dives into the creative interpretations and ideas inspired by the video, rather than summarizing its content. "
                    "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                    "Your task is to first play the role of a human who asks creative questions related to a video and then play the role of an AI assistant that provides imaginative responses based on the video content."
                    "##TASK:"
                    "Users will provide a caption of a video, and you will generate a conversation-like creative question and answer related to the video. "
                    "The question should be designed to explore imaginative aspects of the video, such as creating a story, poem, or alternate scenario inspired by the video. "
                    "You have information about the video based on the provided caption."
                    "Generate ONLY ONE creative questions and detailed answers based on the caption. "
                    "------"
                    "##INSTRUCTIONS:"
                    "- The question must be like a human conversation and inspired by the events in the video. "
                    "- The creative question should prompt for a poem, short story, alternate scenario, or other imaginative response inspired by the video content. "
                    "- The answer must be detailed, descriptive, and imaginative, showcasing creative interpretations of the video. "
                    "------"
                    "##SAMPLE QUESTIONS:"
                    "- Can you write a short poem inspired by the clip?"
                    "- Create a short story that incorporates elements from given game clip."
                    "- How would you turn this game clip into a fairy tale with a moral lesson?"
                    "- Imagine the game clip as a movie scene. How would you describe its climax?"
                    "- Can you create a haiku that captures the essence of given clip?"
                    "- Write a short, suspenseful thriller scene inspired by this game video."
                    "- Write a brief scene from a sci-fi or fantasy novel inspired by given clip.",
                },
                {
                    "role": "user",
                    "content": f"The video caption is: {caption}. "
                    "Please generate the response in the form of a Python JSON dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                    "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                    "Focus on generating ONLY ONE creative question and answer inspired by the video.",
                },
            ]
            completion_2 = openai.ChatCompletion.create(
                messages=message,
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
            )
            # Extract Creative Based QA pairs
            # Convert response to a list of dictionary.
            response_message = completion_2["choices"][0]["message"]["content"]
        
        try:
            response_dict = ast.literal_eval(response_message.replace("\n", ""))
            print("\nresponse_dict: ",json.dumps(response_dict, indent=4))
            # breakpoint()
        except:
            print("Error: Invalid response format. Manual...\n\n")
            print("response_dict= ", response_message)
            breakpoint()
        print("\nresponse_message: ", response_message)
        json_file_path = os.path.join(output_dir, file.split(".")[0] + ".json")
        with open(json_file_path, "w") as f:
            print(
                f"Saving annotations for {key} with caption {caption} in {json_file_path} as _______> {response_dict}"
            )
            json.dump(response_dict, f)
        save_analytics(caption, message, response_message, response_dict)
        print(key)
    print(f"Completed, Annotations saved in {output_dir}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    args = parse_args()

    # Read ground truth captions file
    # list  all files in GPT4-prompt-experiments/e1_captions_tmp

    if args.input_type == "e1":
        args.output_dir = "e1_qa"
        args.input_dir = "e1_captions_tmp"
    elif args.input_type == "e2":
        args.output_dir = "e2_qa"
        args.input_dir = "e2_captions_tmp"

    gt_captions = {}
    for file in os.listdir(args.input_dir):
        with open(f"{args.input_dir}/{file}") as f:
            id = file.split(".")[0]
            gt_captions[id] = json.load(f)

    print(f"len of data: {len(gt_captions)}")

    # Get the video_file_names
    video_files = list(gt_captions.keys())

    caption = {}
    for video_file in tqdm(video_files):
        key = video_file  # Strip file extension.
        try:
            gt_sentences = gt_captions[key]["A"]
        except KeyError:
            print("Warning: GT captions not found for video file. Skipping...")
            continue
        caption[key] = gt_sentences

    # Prepare list of caption files

    caption_files = []
    print(args.creative, args.summary, args.caption, args)
    for video_id in caption.keys():
        if args.creative:
            caption_files.append(f"{video_id}_creative.json")
        if args.summary:
            caption_files.append(f"{video_id}_summary.json")
        if args.caption:
            caption_files.append(f"{video_id}_caption.json")

    print(f"len of output files: {len(caption_files)}")

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set OpenAI API key
    openai.api_key = json.load(open("tmp/openai.json"))["api-key"]
    num_tasks = args.num_tasks

    # Main loop: Continues until all question-answer pairs are generated for all captions
    while True:
        try:
            # Files that have already been completed.
            completed_files = os.listdir(args.output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                print("All tasks completed!")
                break

            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            num_tasks = min(len(incomplete_files), num_tasks)
            part_len = len(incomplete_files) // num_tasks
            all_parts = [
                incomplete_files[i : i + part_len]
                for i in range(0, len(incomplete_files), part_len)
            ]

            task_args = [(caption, part, args.output_dir, args) for part in all_parts]
            # Use a pool of workers to process the files in parallel.
            # with Pool() as pool:
            #     pool.starmap(annotate, task_args)
            for task_arg in task_args:
                annotate(*task_arg)

        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 2 minutes...")
            time.sleep(120)  # wait for 2 minutes before trying again


if __name__ == "__main__":
    main()


# python generate_instruction_qa_human_assisted.py --input_type e2  --summary  1 --caption 1 --creative 1
