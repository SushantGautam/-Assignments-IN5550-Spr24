import glob
import json
import pandas as pd


def all_games_camera_labels():
    all_json_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet/**/Labels-cameras.json", recursive=True)
    list_all_games = []
    for idx, game in enumerate(all_json_files):
            annotations = json.load(open(game))["annotations"]
            game = game.replace("/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet/", "").replace("/Labels-cameras.json", "")
            for annotation in annotations:
                if "link" in annotation:
                    gameTime = annotation["link"]['half'] + " - " + annotation["link"]['time']
                    label, team, visibility = annotation["link"].get("label",None), annotation["link"].get("team",None), annotation["link"].get("visibility",None)
                    list_all_games.append([game, gameTime, label, team, visibility])

    df = pd.DataFrame(list_all_games, columns=['game', 'gameTime', 'label', 'team', 'visibility'])
    df.sort_values(by=['game', 'gameTime', 'label'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['team'] != 'not applicable']
    df = df[df['visibility'] != 'not applicable']
    df = df[df['visibility'] != 'not shown']
    df = df[df['label'] != "I don't know"]
    print(df.groupby(['label', 'team', 'visibility']).size().reset_index(name='count'))
    df.to_csv("all_games_camera_labels.csv", index=False)
    return df

all_games_camera_labels =  all_games_camera_labels()
# all_games_camera_labels = pd.read_csv("all_games_camera_labels.csv")

all_json_files = glob.glob("/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet/**/*Labels-v2.json", recursive=True)

main_df = pd.DataFrame()
for idx, json_file in enumerate(all_json_files):
    annotations = json.load(open(json_file))["annotations"]
    json_file = json_file.replace("/home/sushant/D1/SoccerNetExperiments/Video-LLaMA-SoccerNet/SoccerNet/", "").replace("/Labels-v2.json", "")
    matching_camera_data = all_games_camera_labels[all_games_camera_labels['game'] == json_file]

    for game in annotations:
        # if gameTime in matching_camera_data['gameTime'].values.tolist(), set team in game
        if game['gameTime'] in matching_camera_data['gameTime'].values.tolist():
            matching_game = matching_camera_data[matching_camera_data['gameTime'] == game['gameTime']]
            game['team'] = matching_game['team'].values.tolist()[0]
            game['team'] = matching_game['label'].values.tolist()[0]
            game['visibility'] = matching_game['visibility'].values.tolist()[0]
            # print(game)



    all_game_time = [each["gameTime"] for each in annotations]

    game_time_camera_labels = all_games_camera_labels[all_games_camera_labels['game'] == json_file]['gameTime'].values.tolist()
    # breakpoint()
    # print(idx)
    try:
        assert all(each in all_game_time for each in game_time_camera_labels )
    except AssertionError:
        # print("AssertionError")
        # print(all_game_time)
        # print(game_time_camera_labels)
        print(set(game_time_camera_labels) - set(all_game_time))
        # breakpoint()