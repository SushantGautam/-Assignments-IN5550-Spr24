import glob
import configparser
import cv2
import pandas as pd
import json

# comments_all ={}
# comments = glob.glob("/home/sushant/D1/SoccerNetExperiments/SoccerNetCommentaries/data/**/*.json", recursive=True)
# for comment_file in comments:
#     print(comment_file)
#     comment_key_ = comment_file.replace("/home/sushant/D1/SoccerNetExperiments/SoccerNetCommentaries/data/", "").replace(".json", "")
#     annotations_ = json.load(open(comment_file))["segments"]
#     for annotation in annotations_.values():
#         comment_key = comment_key_
#         annotation_ =[float(annotation[0]), float(annotation[1]), annotation[2]]
#         comments_all.setdefault(comment_key, []).append(annotation_)

# df = pd.DataFrame([(key, val) for key, vals in comments_all.items() for val in vals],
#                   columns=['Game', 'Data'])
# df = pd.concat([df.drop(['Data'], axis=1), df['Data'].apply(pd.Series)], axis=1)
# df.columns = ['Game', 'start', 'end', 'comment']

# breakpoint()
# df['Game']=df['Game'].apply(lambda x: x.replace("/home/sushant/D1/SoccerNetExperiments/SoccerNetCommentaries/data/", "").replace(".json", ""))
# df.to_csv("all_games_comments.csv", index=False)

all_games_captions = pd.read_csv("all_games_captions.csv")
all_games_captions = all_games_captions[all_games_captions.team_identified.notnull()]

all_games_comments = pd.read_csv("all_games_comments.csv")

events_all_= pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQMaaNO_0JU-A2gdSyJpF-WEjJGqWqZdIIp9g9gHGpTdJ3G8l6BvV1PvtmrB3nUTHxnDC_zbiAp3sJx/pub?gid=0&single=true&output=csv')

# events_all = events_all_[events_all_['Pair-label'].notnull()]
# filter sure diff>9
events_all = events_all_[(events_all_['diff'] > 8)]

print(events_all)

t_delta_l = 3
t_delta_r = 10

count_match = []
# randomized_df = events_all.head(40000).sample(frac=1)

for index, row in events_all.iterrows():
    if index % 1000 == 0:
        print("\033[H\033[J")
        print(index)

    start, end, game = row['start'], row['end'], row['game']
    captions = all_games_captions[(all_games_captions['Game'] == game) & 
                              (all_games_captions['gameTime'] >= start+ t_delta_l) & 
                              (all_games_captions['gameTime'] <= end+t_delta_r) ]
    if not captions.empty:
        comments = all_games_comments[(all_games_comments['Game'] == game) & 
                                (all_games_comments['start'] >= start+ 0) & 
                                (all_games_comments['end'] <= end+t_delta_r)]

        captions = captions[captions['team_identified'].notnull()]
        comments = comments[comments['comment'].notnull()]
        events_all.at[index, 'captions'] = " ".join(captions.team_identified.to_list())
        if not comments.empty:
            events_all.at[index, 'comments'] = " ".join(comments.comment.to_list())

    #     print(row.label,"\n", "ASR: ", comments.comment.to_list(), "\n")
    #     print("Caption: ", captions.team_identified.to_list(), "\n\n\n\n\n")
    #     breakpoint()

def remove_repeating_phrases(input_string):
    words = input_string.split()
    unique_words_set = set()
    unique_words_list = []
    for word in words:
        if word not in unique_words_set:
            unique_words_set.add(word)
            unique_words_list.append(word)
    return ' '.join(unique_words_list)


breakpoint()

events_fil = events_all[(events_all['comments'].notna()) & (events_all['replay']==0) & ( events_all['captions'].notna())]
events_fil['comments_len'] = events_all[events_all['comments'].notna()]['comments'].apply(lambda x: len(x.split(" ")))
events_fil['captions_len'] = events_all[events_all['captions'].notna()]['captions'].apply(lambda x: len(x.split(" ")))

events_fil['comments_'] = events_fil.apply(lambda x: remove_repeating_phrases(x['comments']), axis=1)
events_fil['comments_len_'] = events_fil['comments_'].apply(lambda x: len(x.split(" ")))

index_top_not_appplicable = events_fil.sort_values(by=['comments_len_'], ascending=False).head(2000).index
home_away_index = events_fil[events_fil['team'] != 'not applicable'].index
filtered_events_fil = events_fil[events_fil.index.isin(index_top_not_appplicable) | events_fil.index.isin(home_away_index)]

# remove columns  'comments_len', 'captions_len','comments_', 'comments_len_'
filtered_events_fil = filtered_events_fil.drop(['comments_len', 'captions_len','comments_len_'], axis=1)
filtered_events_fil = filtered_events_fil.rename(columns={'comments_': 'comments'})
filtered_events_fil.to_csv("1E+Cap+ASR_10k.csv", index=True)

SN_base = '/home/sushant/D1/DataSets/SoccerNet/'
filtered_events_fil['ffmpeg_code']= filtered_events_fil.apply(lambda r: "ffmpeg  -n  -i '"+SN_base + r.game+"_720p.mkv'  -c:v libx264  -c:a aac -ss "+str(r.start)+" -to "+str(r.end)+" '/home/sushant/D1/MyDataSets/SN_Chunks_1ECapASR_10k/"+str(r.name)+"_"+r.label+".mp4'", axis=1)
filtered_events_fil['ffmpeg_code'].to_csv("1E+Cap+ASR_10k_ffmpeg_code.txt", index=False)


#     if captions.empty:
#         continue
#     matched = captions.team_identified.to_list()
#     # print(game, start, row.curr, end)
#     # print(row.label, matched)
#     count_match.append(len(matched))
#     # breakpoint()
#     try:
#         events_all.at[index, 'captions'] = " ".join(matched)
#     except:
#         breakpoint()

# breakpoint()
# events_all[events_all['captions'].notnull() & ((events_all['team'] == 'home') | (events_all['team'] == 'away'))]
# events_all = events_all[(events_all['team'] == 'home') | (events_all['team'] == 'away')]
# events_all[events_all['captions'].notnull()]['label'].value_counts()