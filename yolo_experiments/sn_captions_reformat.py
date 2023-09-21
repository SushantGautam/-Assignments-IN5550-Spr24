import glob, json
import difflib
import re

caption_root= "/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet-Caption"
jsons = glob.glob(caption_root + "/**/*.json", recursive=True)



def replace_match(match):
    replace_match.counter += 1
    return f"[{match.group(1)}-{replace_match.counter}]"
replace_match.counter = 0

def construct_diff_dict(diff_list):
    diff_dict = {}
    current_key = None
    current_value = []
    current_change = None
    
    for line in diff_list:
        if line.startswith('+ '):
            if current_key is not None:
                diff_dict[current_key] = ' '.join(current_value)
                current_value = []
            current_key = line[2:]
            current_change = '+'
        elif line.startswith('- '):
            current_value.append(line[2:])
            current_change = '-'
        elif line.startswith(' '):
            if current_change is not None:
                if current_key is not None:
                    diff_dict[current_key] = ' '.join(current_value)
                current_key = None
                current_value = []
                current_change = None
    
    return diff_dict

import sys


# json_file_0="/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet-Caption/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-caption.json"
for file in jsons:
    with open(file) as f:
        print(">>>>>>>>> File: "+ file)
        data = json.load(f)
        teams_home, teams_away=  data['home']['names'], data ['away']['names']

        home_player_info = {player["hash"]: "Jersey " +player["shirt_number"]  for player in data['lineup']["home"]["players"]}
        away_player_info = {player["hash"]:  "Jersey " +player["shirt_number"] for player in data['lineup']["away"]["players"]}
        home_tactics = data['lineup']["home"]["tactic"]
        away_tactics = data['lineup']["away"]["tactic"]

        # print(data.keys())
        # inverse order of annotations
        for annotation in reversed(data['annotations']):
            if annotation['label'] in ["attendance",'time', 'whistle', 'substitution'] and annotation['visibility'] == "shown":
                continue
            org_b=annotation['identified']
            org_b = re.sub(r'\[([A-Za-z0-9_]+)\]', lambda match: replace_match(match), org_b)

            a=annotation['description'].replace("(", " ( ").replace(")", " ) ").replace(".", " . ").replace("!", " ! ")+ "_"
            b= org_b.replace("(", " ( ").replace(")", " ) ") .replace(".", " . ").replace("!", " ! ") +"_"

            difference = difflib.Differ()
            diff = list(difference.compare(a.split(), b.split()))

            diff_dict = {}
            diff_dict = construct_diff_dict(diff)

            # print(diff_dict)
            # assert legth of all values in dict is greater than 0
            # assert all([len(v) > 0 for v in diff_dict.values()])
            # for all key starting with TEAM_
            for k,v in diff_dict.items():
                if k.startswith('[TEAM_'):
                    # assert that the value is is teas
                    _v= v.replace(",","").replace(" .", ".")
                    _v= _v.split("'")[0]
                    # print(_v)
                    #_v is either in teams_home or teams_away, return the one that is in the list
                    if _v in teams_home:
                        home_away= "Home team"
                    elif _v in teams_away:
                        home_away= "Away team"
                    else:
                        raise Exception("Team not found in teams list")
                    org_b = org_b.replace(k.replace(" ", ''), home_away)
                elif k.startswith('[PLAYER_'):
                    # print("k: '", k, "'")
                    player_hash = k.split("_")[1].split("-")[0]
                    # assert player_hash in home_player_info or player_hash in away_player_info
                    player_number = home_player_info.get(player_hash) if player_hash in home_player_info else away_player_info.get(player_hash) if player_hash in away_player_info else None
                    org_b = org_b.replace(k.replace(" ", ''), player_number)
                elif k.startswith('[COACH_'):
                    org_b = org_b.replace(k.replace(" ", ''), "Coach")
                elif k.startswith('[REFEREE'):
                    org_b = org_b.replace(k.replace(" ", ''), "Referee")
            print( org_b)
                    




# a = '''Ashley Barnes (Burnley) finds his way to a rebound on the edge of the box, shoots to the top right corner, but the keeper dives and denies him. Burnley have been awarded a corner kick.'''.replace("(", " ( ").replace(")", " ) ") 
# b = '''[PLAYER_zk96WOb5] ([TEAM_]) finds his way to a rebound on the edge of the box, shoots to the top right corner, but the keeper dives and denies him. [TEAM_] have been awarded a corner kick.'''.replace("(", " ( ").replace(")", " ) ") 
# b = re.sub(r'\[([A-Za-z0-9_]+)\]', lambda match: replace_match(match), b)

# difference = difflib.Differ()
# diff = list(difference.compare(a.split(), b.split()))

# diff_dict = {}
# diff_dict = construct_diff_dict(diff)

# print(diff_dict)

