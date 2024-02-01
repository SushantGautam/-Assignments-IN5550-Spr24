import glob

all_captions_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/e*_captions_tmp/*.json")
pkl_paths = "/home/sushant/D1/MyDataSets/SN_Chunks_clip_features/"

import json
import os
all_big_captions = []

for file in all_captions_files:
    file_id = file.split("/")[-1].split(".")[0]
    pkl_id = file_id + ("_1E" if "/e1" in file else "_2E")
    pkl_path = pkl_paths +pkl_id + ".pkl"
    # print(pkl_paths)
    if not os.path.exists(pkl_path):
        print("pkl not found: ", pkl_path)
        continue
    with open(file) as f:
        data = json.load(f)
        all_big_captions.append({
            "id": pkl_id,
            "video": pkl_id + ".pkl",
            "conversations": [{
                "from": "human",
                "value": data["Q"] + "\n<video>"
            },
            {
                "from": "gpt",
                "value": data["A"]
            }]
        })
print("total big captions: ", len(all_big_captions))


all_creative_captions =[]
all_creative_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/e*/*_creative.json")

for file in all_creative_files:
    file_id = file.split("/")[-1].split(".")[0].split("_")[0]
    pkl_id = file_id + ("_1E" if "/e1" in file else "_2E")
    pkl_path = pkl_paths +pkl_id + ".pkl"
    # print(pkl_paths)
    if not os.path.exists(pkl_path):
        print("pkl not found: ", pkl_path)
        continue
    with open(file) as f:
        data = json.load(f)
        all_creative_captions.append({
            "id": pkl_id,
            "video": pkl_id + ".pkl",
            "conversations": [{
                "from": "human",
                "value": data["Q"] + "\n<video>"
            },
            {
                "from": "gpt",
                "value": data["A"]
            }]
        })

print("total creative captions: ", len(all_creative_captions))





all_summary_captions =[]
all_summary_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/e*_qa/*_summary.json")
for file in all_summary_files:
    file_id = file.split("/")[-1].split(".")[0].split("_")[0]
    pkl_id = file_id + ("_1E" if "/e1" in file else "_2E")
    pkl_path = pkl_paths +pkl_id + ".pkl"
    # print(pkl_paths)
    if not os.path.exists(pkl_path):
        print("pkl not found: ", pkl_path)
        continue
    with open(file) as f:
        data = json.load(f)
        # filter  keys in data starting with "Q"
        Q_keys = [key for key in data.keys() if key.startswith("Q")]    
        for q_key in Q_keys:
            a_key = q_key.replace("Q", "A")
            all_summary_captions.append({
                "id": pkl_id,
                "video": pkl_id + ".pkl",
                "conversations": [{
                    "from": "human",
                    "value":  data[q_key] + "\n<video>"
                },
                {
                    "from": "gpt",
                    "value": data[a_key]
                }]
            })
print("total summary captions: ", len(all_summary_captions)) 



all_small_captions =[]
all_small_caption_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/e*_qa/*_caption.json")
for file in all_small_caption_files:
    file_id = file.split("/")[-1].split(".")[0].split("_")[0]
    pkl_id = file_id + ("_1E" if "/e1" in file else "_2E")
    pkl_path = pkl_paths +pkl_id + ".pkl"
    # print(pkl_paths)
    if not os.path.exists(pkl_path):
        print("pkl not found: ", pkl_path)
        continue
    with open(file) as f:
        data = json.load(f)
        # filter  keys in data starting with "Q"
        Q_keys = [key for key in data.keys() if key.startswith("Q")]    
        for q_key in Q_keys:
            a_key = q_key.replace("Q", "A")
            all_small_captions.append({
                "id": pkl_id,
                "video": pkl_id + ".pkl",
                "conversations": [{
                    "from": "human",
                    "value":  data[q_key] + "\n<video>"
                },
                {
                    "from": "gpt",
                    "value": data[a_key]
                }]
            })

print("total small summary captions: ", len(all_small_captions))

# save as json
all_captions_total = all_big_captions + all_creative_captions + all_summary_captions + all_small_captions

print("total captions: ", len(all_captions_total))
with open("/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/tmp/video_chatgpt_training_gen.json", "w") as f:
    json.dump(all_captions_total, f, indent=4)

# python3 /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/tmp/create_video_chatgpt_training.py

# venv/bin/torchrun  --nproc_per_node=5 --master_port 29001 video_chatgpt/train/train_mem.py \
#           --model_name_or_path /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
#           --version v1 \
#           --data_path /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/tmp/video_chatgpt_training_gen.json \
#           --video_folder /home/sushant/D1/MyDataSets/SN_Chunks_clip_features \
#           --tune_mm_mlp_adapter True \
#           --mm_use_vid_start_end \
#           --bf16 True \
#           --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints_mm \
#           --num_train_epochs 3 \
#           --per_device_train_batch_size 8 \
#           --per_device_eval_batch_size 8 \
#           --gradient_accumulation_steps 1 \
#           --evaluation_strategy "no" \
#           --save_strategy "steps" \
#           --save_steps 3000 \
#           --save_total_limit 3 \
#           --learning_rate 2e-5 \
#           --weight_decay 0. \
#           --warmup_ratio 0.03 \
#           --lr_scheduler_type "cosine" \
#           --logging_steps 100 \
#           --tf32 True \
#           --model_max_length 2048 \
#           --gradient_checkpointing True \
#           --lazy_preprocess True




# venv/bin/python video_chatgpt/demo/video_demo.py \
#           --model-name /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
#          --projection_path /home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/Video-ChatGPT_7B-1.1_Checkpoints_mm/mm_projector/checkpoint-3000.bin