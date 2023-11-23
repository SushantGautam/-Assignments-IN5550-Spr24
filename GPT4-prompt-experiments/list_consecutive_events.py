# get all Labels-v2.json files from /home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet
import glob
import json
import pandas as pd


def game_time_to_seconds(game_time):
    half, game_time = game_time.split(" - ")
    minutes, seconds = game_time.split(":")
    return int(half)* 10000 +int(minutes)*60 + int(seconds)

all_json_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet/**/*Labels-v2.json", recursive=True)

main_df = pd.DataFrame()
for idx, json_file in enumerate(all_json_files):
    annotations = json.load(open(json_file))["annotations"]
    df = pd.DataFrame(annotations)
    df = df[df['visibility'] == 'visible']
    df["start_seconds"] = df["gameTime"].apply(game_time_to_seconds)
    df['Event2']  = df['label'].shift(-1)
    df['timeDiff'] = df['start_seconds'].shift(-1) - df['start_seconds']
    main_df = pd.concat([main_df, df], ignore_index=True)

event_pairs = main_df.dropna(subset=['Event2']).loc[(main_df['timeDiff'] >= 2) & (main_df['timeDiff'] <= 7)]

recurring_pairs = (
    event_pairs.groupby(['label', 'Event2'])
           .size()
           .reset_index(name='Frequency')
           .query('Frequency > 1')
           .sort_values(by='Frequency', ascending=False)
)

print(recurring_pairs)
breakpoint()
