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

all_games_captions = pd.read_csv("all_games_captions.csv")
# breakpoint()


events_all_ = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=464577990&single=true&output=csv')
events_all = events_all_[events_all_['Pair-label'].notnull()]
t_delta_l = 3
t_delta_r = 10

count_match = []
for index, row in events_all.iterrows():
    start, end, game = row['pair_start']+ t_delta_l, row['pair_end']+t_delta_r, row['game']
    captions = all_games_captions[(all_games_captions['Game'] == game) & 
                              (all_games_captions['gameTime'] >= start) & 
                              (all_games_captions['gameTime'] <= end)]
    # make sure they are not empty
    captions = captions[captions['team_identified'].notnull()]

    print(index)
    if captions.empty:
        continue
    matched = captions.team_identified.to_list()
    # print(row.label, events_all.iloc[index + 1].label, matched)
    count_match.append(len(matched))
    # breakpoint()
    try:
        events_all.at[index, 'captions'] = " ".join(matched)
    except:
        breakpoint()

# save to csv
# events_all.to_csv('event_pairs_with_captions.csv', index=False)