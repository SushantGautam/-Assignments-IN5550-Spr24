# jsons_dir= "/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa"

# combined_json_output_dir = "/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa_json"

# # need three JSON files, in combined_json_output_dir
# # temporal_qa.json, consistency_qa.json, and generic_qa.json

# # jsons_dir contains mixture of JSON files, which are used to generate the three JSON files above
# <id>_detail.json, <id>_temporal.json, <id>_consistency.json


# # /home/sushant/D1/MyDataSets/SN_Chunks_2Events_10s contains videos starting with <id>
# #create a dict with key as <id> and value as the video file name without .mp4 (not path)
    
# format of final:


# temporal_qa.json
# [{"Q": "...", "A": "..", "video_name": ".."}, {"Q": "...", "A": "..", "video_name": ".."}..]

# consistency_qa.json
# [{"Q1": "...", "Q2",:"...", "A": "..", "video_name": ".."}, {"Q1": "...", "Q2",:"...", "A": "..", "video_name": ".."}..]

# generic_qa.json
# [{"Q": "...", "A": "..", "video_name": ".."}, {"Q": "...", "A": "..", "video_name": ".."}..]


import os
import json

def read_json_files(directory):
    files_content = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                files_content[filename] = json.load(file)
    return files_content

def create_video_dict(videos_directory):
    video_dict = {}
    for video_name in os.listdir(videos_directory):
        if video_name.endswith('.mp4'):
            key = video_name.split('_')[0]
            video_dict[key] = video_name.rsplit('.', 1)[0]
    return video_dict

def combine_jsons(files_content, video_dict):
    temporal_qa, consistency_qa, generic_qa = [], [], []
    
    for filename, content in files_content.items():
        base_name = filename.split('_')[0]
        video_name = video_dict.get(base_name, "")
        
        if "_temporal.json" in filename:
                content.update({"video_name": video_name})
                temporal_qa.append(content)
        elif "_consistency.json" in filename:
                content.update({"video_name": video_name})
                consistency_qa.append(content)
        elif "_detail.json" in filename:
                content.update({"video_name": video_name})
                generic_qa.append(content)
                
    return temporal_qa, consistency_qa, generic_qa

def save_json(data, path):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Adjust these paths as needed
jsons_dir = "/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa"
videos_dir = "/home/sushant/D1/MyDataSets/SN_Chunks_2Events_10s"
output_dir = "/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa_json"


files_content = read_json_files(jsons_dir)
video_dict = create_video_dict(videos_dir)
temporal_qa, consistency_qa, generic_qa = combine_jsons(files_content, video_dict)

# Save the combined JSON files
save_json(temporal_qa, os.path.join(output_dir, "temporal_qa.json"))
save_json(consistency_qa, os.path.join(output_dir, "consistency_qa.json"))
save_json(generic_qa, os.path.join(output_dir, "generic_qa.json"))

# cd ~/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments
# python generate_benchmark_qa_json.py 

# then:
# cd ~/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/
# python video_chatgpt/eval/run_inference_benchmark_general.py \
#     --video_dir  "/home/sushant/D1/MyDataSets/SN_Chunks_2Events_10s" \
#     --gt_file /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa_json/generic_qa.json \
#     --output_dir /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/benchmark_qa_json \
#     --output_name generic_qa_pred \
#     --model-name /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
#     --projection_path /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/Video-ChatGPT_7B-1.1_Checkpoints_mm/mm_projector.bin