import spacy

# Load the English language model from spaCy
nlp = spacy.load("en_core_web_lg")

import glob
import json

all_json_files = glob.glob(
    "/home/sushant/D1/SoccerNetExperiments/Soccer-Video-ChatGPT/GPT4-prompt-experiments/e2_captions_tmp/*.json"
)
print(len(all_json_files))
count = 0
for idx, file in enumerate(all_json_files):
    print(file)
    with open(file) as f:
        text = json.load(f)["A"]
    # Process the text with spaCy
    doc = nlp(text)

    # Extract player names and team names
    player_names = []
    team_names = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            player_names.append(ent.text)
        elif ent.label_ == "ORG" and ent.text != "Referee" and 'jersey'not in ent.text.lower() and '/' not in ent.text:
            team_names.append(ent.text)

    if len(player_names) > 0 or len(team_names) > 0:
        print(text)
        print(file)
        print(player_names, team_names)
        breakpoint()
        count += 1
        print("total: ", count, "of", idx)

