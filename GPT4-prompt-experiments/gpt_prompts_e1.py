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
E1_CapASR_10k = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=353977511&single=true&output=csv', index_col=0)


def describe_football_clipE1(events):
    # Extracting event information
    event1, event1_team, home_jersey, away_jersey, caption, commentary = events

    # Mapping team to their jersey color
    team_color = {'home': home_jersey, 'away': away_jersey, 'not applicable': 'none'}

    # Describing the first event
    if event1_team != 'not applicable':
        description1 = f"{event1} event by {team_color[event1_team]}-jerseyed team"
    else:
        description1 = f"{event1} event"

    # Combining the descriptions and adding context about the teams
    full_description = (f"Video Clip: \n shows only {description1} in the match between teams in {home_jersey} vs  {away_jersey} jerseys."
                       )
    if caption:
        _caption= caption.replace('Away-Team', away_jersey + "-jerseyed team").replace('Home-Team', home_jersey + "-jerseyed team")
        full_description += f"\n-----\n Possible Supporting Caption: \n{_caption}"
        full_description += f"\n-----\n Possible Supporting Commentary: \n{commentary}"

    return full_description

import os 
import json

count = 0
all_output = []

E1_CapASR_10k = E1_CapASR_10k.iloc[:1000]

for index, row in E1_CapASR_10k.iterrows():
    if os.path.isfile(f"e1_captions_tmp/{index}.json"):
        print(f"e1_captions_tmp/{index}.json exists")
    else:
        # print(index, row['captions'])
        formatted_cap = describe_football_clipE1(events = [row.label,  row.team, row['home_color'], row['away_color'], row.captions, row.comments_ASR])
        print(index, formatted_cap)
        gpt_response_json = call_gpt4(formatted_cap, event_n=1)
        # breakpoint()
        json.dump(gpt_response_json, open(f"e1_captions_tmp/{index}.json", 'w'))
        count += 1

# # all_output to csv
# all_output_df = pd.DataFrame(all_output, columns=['index', 'GPT-response', 'event1', 'team1', 'event2', 'team2', 'home_color', 'away_color', 'captions', ])
# all_output_df.to_csv('gpt_all_output1.csv', index=False)
