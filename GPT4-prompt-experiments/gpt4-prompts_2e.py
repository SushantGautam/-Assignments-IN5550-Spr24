# import glob
# import configparser
# import cv2
# import pandas as pd

# test = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQayMZ1WdRur-ICBFw7QQnvaZytK5PalxLdvtObR08wR2OPvSS3QNmPCPT372Nu4odVZePD8AzzOH-N/pub?gid=362554130&single=true&output=csv',
#                    index_col=0,
#                   )


# pattern = "/home/sushant/D1/DataSets/SoccerNet-tracking/tracking-2023/test/SNMOT-*/gameinfo.ini"

# # Use glob to find all matching files
# matching_files = glob.glob(pattern)


# for ini_file_path in matching_files:
#     trackletID_values = {}
#     config = configparser.ConfigParser()
#     config.read(ini_file_path)
#     if 'actionClass' in config['Sequence']:
#         if 'actionclass' in config['Sequence']:
#             for key, value in config['Sequence'].items():
#                 if key.startswith('trackletid_'):
#                     trackletID_values[int(key.replace("trackletid_", ''))] = value
#         trackletID_values['action'] = config['Sequence']['actionclass']
#         trackletID_values['gameID'] = config['Sequence']['gameid']
#         all_games_trackID[ini_file_path] = trackletID_values
    
#     gt_file_path = ini_file_path.replace("gameinfo.ini", "gt/gt.txt")
#     with open(gt_file_path) as f:
#         lines = f.read().splitlines()
#         all_games_tracking[ini_file_path] = lines
        
# # for k,v in all_games.items():
# #     print(k+ ","+ v )


#read csv
import glob
import configparser
import pandas as pd
from gpt4_token_experiment import call_gpt4 

# captions 
all_data= pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=957600063&single=true&output=csv", index_col=0)
# find all rows with two -> in pair label
# two_events = all_data[all_data['Pair-label'].str.contains('->Ye')]['Pair-label'].unique()
# # e2_captions = all_data[(all_data['comb_tpe'] == 'Cap') & all_data['gpt4'].isna()]
e2_captions = all_data[(all_data['comb_tpe'] == 'Cap+Com')]

def describe_football_clip(events):
    # Extracting event information
    event1, event1_team, event2, event2_team, home_jersey, away_jersey, caption, comments = events

    # Mapping team to their jersey color
    team_color = {'home': home_jersey, 'away': away_jersey, 'not applicable': 'none'}

    # Describing the first event
    if event1_team != 'not applicable':
        description1 = f"First, The {team_color[event1_team]}-jerseyed team executes {event1}."
    else:
        description1 = f"Initially, {event1} occurs."

    # Describing the second event
    if event2_team != 'not applicable':
        description2 = f"Following that, the {team_color[event2_team]}-jerseyed team performs {event2} "
    else:
        description2 = f"Subsequently, {event2} takes place "

    # Combining the descriptions and adding context about the teams
    full_description = (f"Video:\nshows two events of the match between teams in {home_jersey} and {away_jersey} jerseys, "
                        f"{description1} {description2}and the clip ends.")
    if caption:
        _caption= caption.replace('Away-Team', away_jersey + "-jerseyed team").replace('Home-Team', home_jersey + "-jerseyed team")
        full_description += f"\n Possible Supporting caption: {_caption}"
    if comments and len(comments) > 0:
        full_description += f"\n Possible Supporting commentary: {comments}"
    return full_description

import os 
import json

count = 0
all_output = []
for index, row in e2_captions.iterrows():
    event, n_event = row['Pair-label'].split('->')

    if os.path.isfile(f"e2_captions_tmp/{index}.json"):
        print(f"e2_captions_tmp/{index}.json exists")
        # gpt_response_json = json.load(open(f"e2_captions_tmp/{index}.json"))['A']
        # all_output.append([index,gpt_response_json ])
        continue
    else:
        all_output.append([index,None ])
        if (len(row.comments.split(' ')) - len(row.comments_.split(' ')))/len(row.comments.split(' ')) > 0.3:
            comm = row.comments_
        else:
            comm = row.comments

        formatted_cap = describe_football_clip(events = [event, row.team, n_event, row.n_team, row.home_color, row.away_color, row.captions, comm])
        print([event, row.team, n_event, row.n_team, row.home_color, row.away_color, row.captions])
        print(count, index, formatted_cap)
        gpt_response_json = call_gpt4(formatted_cap, event_n=2)
        # breakpoint()
        json.dump(gpt_response_json, open(f"e2_captions_tmp/{index}.json", 'w'))
        count += 1