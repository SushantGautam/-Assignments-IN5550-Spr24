# Required Libraries
import openai
import os
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm
from multiprocessing.pool import Pool

# Suppressing all warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Descriptive question-answer-generation-using-GPT-3")
    parser.add_argument("--input_type", required=True, help="e1 or e2")   
    parser.add_argument("--num_tasks", required=False, type=int, help="Number of splits.", default=10)
    parser.add_argument("--creative", required=False, type=bool, help="Generate Creative Based QA pairs.", default=False)
    parser.add_argument("--summary", required=False, type=bool, help="Generate Summary Based QA pairs.", default=False)
    parser.add_argument("--caption", required=False, type=bool, help="Generate Caption Based QA pairs.", default=False)

    return parser.parse_args()


def annotate(gt_file, caption_files, output_dir, args):
    """
    Generates question and answer pairs based on video captions using OpenAI GPT-3.
    """
    for file in tqdm(caption_files):
        key = file.split(".")[0].split("_")[0]
        caption = gt_file[key]

        #tmp #########

        # if file.split(".")[0].split("_")[1] == 'summary':
        #     response_dict = {'summarrt_qa': 'summary_qa'}
        # elif file.split(".")[0].split("_")[1] == 'caption':
        #     response_dict = {'caption_qa': 'caption_qa'}
        # elif file.split(".")[0].split("_")[1] == 'creative':
        #     response_dict = {'creative_qa': 'creative_qa'}

        #end tmp #####


        # Generate QA pairs with OpenAI GPT-3: Summarization
        if file.split(".")[0].split("_")[1] == 'summary':
            completion_0 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                response_format={ "type": "json_object" },
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You play two roles: a human asking questions related to summarizing a video and an intelligent chatbot designed for video summarization and dense captioning. "
                            "Your task is video summarization. "
                            "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                            "Your task is to play the role of a human who asks three questions related to summarizing the video and then play the role of an AI assistant that provides paraphrased answers based on the video content and the provided caption."
                            "------"
                            "##TASK:"
                            "Users will provide a caption of a video, and you will generate a set of three conversation-like questions related to summarizing the video. "
                            "The questions and answers can be very similar, but they should all focus on summarizing the video content. "
                            "The answers should be paraphrased versions of the provided caption. "
                            "You have information about the video based on the provided caption and have summarized the events in it."
                            "Generate THREE different questions asking to summarize the video and provide detailed answers to each based on the caption. "
                            "------"
                            "##INSTRUCTIONS:"
                            "- The questions must be like a human conversation and focused on summarizing the video. "
                            "- The answers must be paraphrased versions of the provided caption, and they should be detailed and descriptive. "
                            "------"
                            "##SAMPLE QUESTIONS:"
                            "- Can you provide a summary of the video?"
                            "- What are the main events in the video?"
                            "- Could you briefly describe the video content?"
                    },
                    {
                        "role": "user",
                        "content":
                            f"The video caption is: {caption}. "
                            "Please generate the response in the form of a Python JSON list of dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                            "For example, your response should look like this: [{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, {'Q': 'Your second question here...', 'A': 'Your second answer here...'}, {'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
                            "Emphasize that the questions and answers can be very similar, but they should all focus on summarizing the video content."
                    }
                ]
            )
            # Extract Summary Based QA pairs
            # Convert response to a list of dictionary.
            response_message_0 = completion_0["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message_0)
        elif file.split(".")[0].split("_")[1] == 'caption':
            # Generate QA pairs with OpenAI GPT-3: Caption Based
            # Answers specifically restricted to information in the caption
            completion_1 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                response_format={ "type": "json_object" },
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You play two roles: a human asking questions related to a video and an intelligent chatbot designed to help people find information from a given video. "
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
                            "- How does the person repair the car?"
                    },
                    {
                        "role": "user",
                        "content":
                            f"The video caption is: {caption}. "
                            "Please generate the response in the form of a Python JSON list of dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                            "For example, your response should look like this: [{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, {'Q': 'Your second question here...', 'A': 'Your second answer here...'}, {'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
                            "Emphasize that the ALL THREE questions must be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers, and provide detailed and descriptive answers."
                    }
                ]
            )
            # Extract Caption Based QA pairs
            # Convert response to a list of dictionary.
            response_message_1 = completion_1["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message_1)
        elif file.split(".")[0].split("_")[1] == 'creative':
            # Generate QA pairs with OpenAI GPT-3: Creative Based
            # TODO: Limit to samples with lengthy GT captions
            completion_2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                response_format={ "type": "json_object" },
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You play two roles: a human asking questions related to a video and an intelligent chatbot designed to help people find information from a given short soccer game clip. "
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
                            "- Write a brief scene from a sci-fi or fantasy novel inspired by given clip."
                    },
                    {
                        "role": "user",
                        "content":
                            f"The video caption is: {caption}. "
                            "Please generate the response in the form of a Python JSON dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                            "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                            "Focus on generating ONLY ONE creative question and answer inspired by the video."
                    }
                ]
            )
            # Extract Creative Based QA pairs
            # Convert response to a list of dictionary.
            response_message_2 = completion_2["choices"][0]["message"]["content"]
            # response_message_2 = response_message_2.replace("{'Q': '", '{"Q": "').replace("', 'A': '", '", "A": "').replace("'}", '"}')
            print(response_message_2)
            response_dict = ast.literal_eval(response_message_2)
        
            
        json_file_path = os.path.join(output_dir, file.split(".")[0]+".json")
        with open(json_file_path, "w") as f:
            print(f"Saving annotations for {key} with caption {caption} in {json_file_path} as _______> {response_dict}")
            json.dump(response_dict, f)

    print(f"Completed, Annotations saved in {output_dir}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    args = parse_args()

    # Read ground truth captions file
    # list  all files in GPT4-prompt-experiments/e1_captions_tmp

    if args.input_type == 'e1':
        args.output_dir = 'e1_qa'
        args.input_dir = 'e1_captions_tmp'
    elif args.input_type == 'e2':
        args.output_dir = 'e2_qa'
        args.input_dir = 'e2_captions_tmp'
    
    gt_captions= {}
    for file in os.listdir(args.input_dir):
        with open(f"{args.input_dir}/{file}") as f:
            id = file.split(".")[0]
            gt_captions[id] = json.load(f)

    print(f"len of data: {len(gt_captions)}")

    # Get the video_file_names
    video_files = list(gt_captions.keys())

    caption = {}
    for video_file in tqdm(video_files):
        key = video_file # Strip file extension.
        try:
            gt_sentences = gt_captions[key]['A']
        except KeyError:
            print(f"Warning: GT captions not found for video file. Skipping...")
            continue
        caption[key] = gt_sentences

    # Prepare list of caption files
            
    caption_files = []
    print(args.creative, args.summary, args.caption, args)
    for video_id in caption.keys():
        if args.creative:
            caption_files.append(f'{video_id}_creative.json')
        if args.summary:
            caption_files.append(f'{video_id}_summary.json')
        if args.caption:
            caption_files.append(f'{video_id}_caption.json')

    print(f"len of output files: {len(caption_files)}")

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set OpenAI API key
    openai.api_key = json.load(open('tmp/openai.json'))["api-key"]
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
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]

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