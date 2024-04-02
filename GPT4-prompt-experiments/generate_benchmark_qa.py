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
    parser.add_argument(
        "--num_tasks", required=False, type=int, help="Number of splits.", default=1
    )
    parser.add_argument(
        "--detail",
        required=False,
        type=bool,
        help="Generate detail Based QA pairs.",
        default=False,
    )
    parser.add_argument(
        "--temporal",
        required=False,
        type=bool,
        help="Generate temporal Based QA pairs.",
        default=False,
    )
    parser.add_argument(
        "--consistency",
        required=False,
        type=bool,
        help="Generate consistency Based QA pairs.",
        default=False,
    )

    return parser.parse_args()

# https://docs.google.com/spreadsheets/d/1yKPA1ZyISFJZtb8NvYbpXq0JLc2aBnB20J8mi2KJOYk/edit#gid=1648738006
e2_data = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSK-oe_IfipTwI_OXRugLStJr3HcmTgBQccwVmwirbzxgUO_67QKOsy7YNDdujrc7IVqRqF2fbqau9V/pub?gid=1648738006&single=true&output=csv", index_col=27, encoding='utf8')


import re
def annotate(caption_files, output_dir, args):
    """
    Generates question and answer pairs based on video captions using OpenAI GPT-3.
    """

    for file in tqdm(caption_files):
        key = file.split(".")[0].split("_")[0]
        row = e2_data.loc[int(file.split('_')[0])]
        comments = row.comments
        event1, event2 = row["Pair-label"].split("->")
        event_info ="Only two events are shown: "+ event1 + " and then " + event2 + ". "
        event_1_team, event_2_team =  row.team, row.n_team
        match = re.search(r'(\d{4}-\d{2}-\d{2} - \d{2}-\d{2}) ([^0-9]+) (\d) - (\d) ([^/]+)/\d+', row.game)
        home_team, away_team = match.group(2).strip(), match.group(5).strip()
        home_color, away_color = row.home_color, row.away_color
        colour = {'home': home_color, 'away': away_color}
        event1_colour, event2_colour = colour.get(event_1_team, None), colour.get(event_2_team, None)


        #event description
        description = "Given a soccer clip with two consecutive visible events:\n 1. "+ event1 + (" by "+ event_1_team +" team in "+ event1_colour +" jersey" if event1_colour else "") + "\n 2. "+ event2 + (" by "+ event_2_team +" team in "+ event2_colour + " jersey" if event2_colour else "") + ".\n"
        # Anonymize Real Madrid (home team) and its players as white jersey team/player and Barcelona  (away team) as blue/red stripe jersey team/player.
        description_team = "Anonymize " + home_team + " (home team) and its players as " + home_color + " jersey team/player and " + away_team + " (away team) and its players as " + away_color + " jersey team/player."

        # print(description)
        # print(description_team)
        caption = description + description_team  + '\n\n The Internal commentary (contain team/player names): "' + comments + '"\n'
        # continue
    
        
        if file.split(".")[0].split("_")[1] == "detail":
            message = [
                    {
                        "role": "system",
                        "content": 
                            # "You will play two roles: a human asking questions related to describing a short socer video clip and an intelligent privacy-preserving chatbot designed for video event description and dense captioning without mentioning real team/player names. You also have access to internal commentary around the clip which is not actually shown in the clip. "
                            # "Your task is to generate a detailed and descriptive paragraph based on the provided fragmented information about a video. You can use commentary to enhance the event understanding. Be aware that the commentary can contain additional information not shown in the clip and not related to events. Don't use real player and team names, strictly anonymize them to prevent privacy."
                            # "------"
                            "##TASK:"
                            "Users will provide information about a short soccer video clip and two visible events, and you will generate ONE conversation-like question and answer related to describing game events in detail. "
                            "The question should ask to describe the video content in detail. We also have access to internal commentary which can enhance the event understanding only if it contains information directly related to the visible events. Other information not directly related to visible events in the commentary should be ignored. "
                            "Don't mention about commentary in question/answer as it is not shown on video."
                            "Be serious about privary and anonymization. Don't use real player and team names, strictly anonymize them to prevent privacy."
                            "Always use jersey colours (eg: player form .. color team,) to refer to team/player as the names are not visible on the video."
                            "The answer should be a paraphrased and well-structured paragraph only based on the provided information, as detailed as possible. "
                    },
                    {
                        "role": "user",
                        "content":
                            f"The user input is: {caption}. "
                            f"Please generate the response in the form of a Python JSON dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                            "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                            f"Emphasize that the answer should focus on describing the video content as detailed as possible without mentioning any player name and team names as well as about the internal commentary."
                    }
                ]
            
            # s = json.dumps(message, indent=1,ensure_ascii=False)
            # print(s)
            # breakpoint()
            # continue
            completion_0 = openai.ChatCompletion.create(
                messages=message,
                model=model_name,
                response_format={"type": "json_object"},
            )
            # Extract Summary Based QA pairs
            # Convert response to a list of dictionary.
            # breakpoint()
            response_message = completion_0["choices"][0]["message"]["content"]
            # response_message ='{"s": "test"}' #REM

        elif file.split(".")[0].split("_")[1] == "temporal":
            message = [
                {
                    "role": "system",
                    "content": "You play two roles: a human asking questions related to a video and an intelligent chatbot designed to help people find information from a given video. "
                    "Your task is video summarization, which will be used by users to understand different events in long videos by asking different questions based on the video. "
                    "The video summarization will be used for various applications such as surveillance, generate previews or summaries of video content for video search engines, "
                    "create highlights or summaries of sporting events, TV shows, and movies. "
                    "Your task is to first play the role of a human who asks questions related to a video and then play the role of an AI assistant that provides information based on the video content."
                    "------"
                    "##TASK:"
                    "Users will provide some information about a video, and you will generate a set of conversation-like questions and answers related to the video. "
                    "The questions should be designed to extract information directly from the given information, so that the provided information or parts of it can serve as the answers. "
                    "Generate THREE different descriptive and conversational style questions and detailed answers based on the given information. "
                    "------"
                    "##INSTRUCTIONS:"
                    "- The questions must be like a human conversation and based on the events in the video. "
                    "- The questions should be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers. "
                    "- The answers must be detailed and descriptive, and they should directly reference the information provided. "
                    "- The questions can be related to the appearance, motion, trajectory, and reasoning. "
                    "------"
                    "##SAMPLE QUESTIONS:"
                    "- What is the man doing in the video?"
                    "- What are the girls doing in the video?"
                    "- Describe the appearance of the motorbike"
                    "- Is the person riding the bike wearing a helmet?"
                    "- How does the person repair the car?",
                },
                {
                    "role": "user",
                    "content": f"The video caption is: {caption}. "
                    "Please generate the response in the form of a Python JSON list of dictionaries with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                    "For example, your response should look like this: [{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, {'Q': 'Your second question here...', 'A': 'Your second answer here...'}, {'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
                    "Emphasize that the ALL THREE questions must be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers, and provide detailed and descriptive answers.",
                },
            ]

            completion_1 = openai.ChatCompletion.create(
                messages=message,
                model=model_name,
                response_format={"type": "json_object"},
            )

            # Extract Caption Based QA pairs
            # Convert response to a list of dictionary.
            response_message = completion_1["choices"][0]["message"]["content"]

        elif file.split(".")[0].split("_")[1] == "consistency":
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
                model=model_name,
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
    args.output_dir = "benchmark_qa"

    caption_files = []
    print(args.detail, args.temporal, args.consistency, args)
    
    for row in e2_data.iterrows():
        video_id = row[0]
        if args.detail:
            caption_files.append(f"{video_id}_detail.json")
        if args.temporal:
            caption_files.append(f"{video_id}_temporal.json")
        if args.consistency:
            caption_files.append(f"{video_id}_consistency.json")

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

            task_args = [(part, args.output_dir, args) for part in all_parts]
            # Use a pool of workers to process the files in parallel.
            # with Pool() as pool:
            #     pool.starmap(annotate, task_args)
            for task_arg in task_args:
                annotate(*task_arg)

        except Exception as e:
            breakpoint()
            print(f"Error: {e}")
            print("Sleeping for 2 minutes...")
            time.sleep(120)  # wait for 2 minutes before trying again


if __name__ == "__main__":
    main()


# python generate_benchmark_qa.py --detail  1 --temporal 1 --consistency 1