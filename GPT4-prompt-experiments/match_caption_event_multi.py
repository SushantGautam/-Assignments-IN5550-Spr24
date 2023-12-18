import glob
import configparser
import cv2
import pandas as pd
import json

# captions_all ={}

# captions = glob.glob("/home/sushant/D1/MyDataSets/SN_Captions_Mod/**/Labels-caption.json", recursive=True)
# for caption_file in captions:
#     print(caption_file)
#     caption_key_ = caption_file.replace("/home/sushant/D1/MyDataSets/SN_Captions_Mod/", "").replace("Labels-caption.json", "")
#     annotations_ = json.load(open(caption_file))["annotations"]
#     annotations = reversed(annotations_)
#     for annotation in annotations:
#         caption_key = caption_key_ + annotation['gameTime'][0]
#         if annotation['gameTime'][0] not in ['1', '2'] or annotation['visibility'] != 'shown' or   annotation['label'] in ['funfact', 'attendance', 'time',]:
#             continue
#         captions_all.setdefault(caption_key, []).append(annotation)

# df = pd.DataFrame([(key, val) for key, vals in captions_all.items() for val in vals],
#                   columns=['Game', 'Data'])
# df = pd.concat([df.drop(['Data'], axis=1), df['Data'].apply(pd.Series)], axis=1)
# df['gameTime'] = df['gameTime'].apply(lambda x: x.split(' - ')[1]) 
# df['gameTime'] = df['gameTime'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
# df['gameTime'] = df['gameTime'].astype('float32')
# df.to_csv("all_games_captions.csv", index=False)

# all_games_captions = pd.read_csv("all_games_captions.csv")
# # breakpoint()


# events_all_ = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=464577990&single=true&output=csv')
# events_all = events_all_[events_all_['Pair-label'].notnull()]
# t_delta_l = 3
# t_delta_r = 10

# count_match = []
# for index, row in events_all.iterrows():
#     start, end, game = row['pair_start']+ t_delta_l, row['pair_end']+t_delta_r, row['game']
#     captions = all_games_captions[(all_games_captions['Game'] == game) & 
#                               (all_games_captions['gameTime'] >= start) & 
#                               (all_games_captions['gameTime'] <= end)]
#     # make sure they are not empty
#     captions = captions[captions['team_identified'].notnull()]

#     print(index)
#     if captions.empty:
#         continue
#     matched = captions.team_identified.to_list()
#     # print(row.label, events_all.iloc[index + 1].label, matched)
#     count_match.append(len(matched))
#     # breakpoint()
#     try:
#         events_all.at[index, 'captions'] = " ".join(matched)
#     except:
#         breakpoint()


# # events_all without filter same id from events_all_
# events_all = events_all_.


def remove_repeating_phrases(input_string):
    words = input_string.split()
    unique_words_set = set()
    unique_words_list = []
    for word in words:
        if word not in unique_words_set:
            unique_words_set.add(word)
            unique_words_list.append(word)
    return ' '.join(unique_words_list)




_2e_captions = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=957600063&single=true&output=csv", index_col=0)
_2e_captionsx = _2e_captions[(_2e_captions.captions.notna()) & (_2e_captions.captions.str.len() > 1)]

all_games_comments = pd.read_csv("all_games_comments.csv")
t_delta_r = 10


home_away_= pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQayMZ1WdRur-ICBFw7QQnvaZytK5PalxLdvtObR08wR2OPvSS3QNmPCPT372Nu4odVZePD8AzzOH-N/pub?gid=752115945&single=true&output=csv')
condition = ~home_away_['T1-color'].astype(str).str.contains('--')
home_away = home_away_[home_away_['T1-color'].notna() & condition]

home_away_dict =  home_away.set_index('Game')[['T1-color', 'T2-color']].to_dict(orient='index')

allevents_sheet1= pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=0&single=true&output=csv")
import os

for index, row in _2e_captionsx.iterrows():
    start, end, game = row['pair_start'], row['pair_end'], row['game']
    comments = all_games_comments[(all_games_comments['Game'] == game) & 
                                (all_games_comments['start'] >= start+ 0) & 
                                (all_games_comments['end'] <= end+t_delta_r)]
    next_row = allevents_sheet1.iloc[index+1]
    _2e_captions.at[index, 'n_team'] = next_row.team

    home_color = home_away_dict[row.game[:-2]]['T1-color']
    away_color = home_away_dict[row.game[:-2]]['T2-color']
    _2e_captions.at[index, 'home_color'] = home_color
    _2e_captions.at[index, 'away_color'] = away_color

    try:
        assert len(comments) > 0
        commentary = " ".join(comments.comment.to_list())
        _2e_captions.at[index, 'comments'] = commentary
        _2e_captions.at[index, 'comments_'] = remove_repeating_phrases(commentary)
        _2e_captions.at[index, 'comb_tpe'] = 'Cap+Com'
    except Exception as e:
        _2e_captions.at[index, 'comb_tpe'] = 'Cap'
    if index < 31140:
        # check if /e2_captions_tmp/index.json
        if os.path.isfile(f"e2_captions_tmp/{index}.json"):
            print(f"e2_captions_tmp/{index}.json exists")
            gpt_response_json = json.load(open(f"e2_captions_tmp/{index}.json"))['A']
            _2e_captions.at[index, 'gpt4'] = gpt_response_json
            _2e_captions.at[index, 'comb_tpe'] = 'Cap'

_2e_captions.to_csv("2E+captions1.csv")
