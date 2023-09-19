import csv

import configparser

config = configparser.ConfigParser()
config.read('/home/sushant/D1/DataSets/SoccerNet-tracking/tracking-2023/test/SNMOT-117/gameinfo.ini')
all_config=  config.items('Sequence')
#filter all_config to get only the key, value starting with trackletID_. output dict# all config is list of tuples
filtered_config = {int(k.replace("trackletid_",'')):v.replace(" team ", " team: ") for k,v in all_config if k.startswith('trackletid_')}
# iterate over filtered_config  and update values

for k,v in filtered_config.items():
    if "player" in v:
        filtered_config[k] = v.replace(";", ", jersey: ").replace("player", "Player")
    elif 'ball' in v:
        filtered_config[k] = v.split(";")[0]
    else:
        filtered_config[k] = v.replace(";", " : ")

import  os
os.system('cls' if os.name == 'nt' else 'clear')

# print('''

# You are an AI visual assistant, and you are seeing a single video clip of a football game highlight that particularly shows an offside event.
# What you see are provided with sentences above describing the same video you are looking at. 
# Answer all questions as you are seeing the video.
# Design a conversation between you and a person asking about this video. 
# The answers should be in a tone that a visual AI assistant is seeing the video and answering the question. 
# Ask diverse questions and give corresponding answers.

# Include questions asking about the visual content of the video, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. 
# Dont refer to frame number, game time, seconds, numeric coordinates and and jersey numbers.
# Only include questions that have definite answers:
# (1) one can see the content in the video that the question asks about and can answer confidently;
# (2) one can determine confidently from the video that it is not in the video. 
# Do not ask any question that cannot be answered confidently.
# Also include complex questions that are relevant to the content in the video, for example, asking
# about background knowledge of the objects in the video, asking to discuss about events happening in
# the video, etc. Again, do not ask about uncertain details. Provide detailed answers when answering
# complex questions. For example, give detailed examples or reasoning steps to make the content more
# convincing and well-organized. You can include multiple paragraphs if necessary."
# ''')



print('''
Given the normalized tracking bounding box coordinates (frame: x, y, x, y) of different entities detected in frames, uniformly sampled from a small (200 seconds) short football game highlight clip that particularly shows an offside event at the end.
Act as a game analyst and summarize the event happening in the clip in natural text as if you are watching a game video clip.
Describe the visual content of the video, including the object types, counting the objects, object actions, object locations, relative positions between players balls, keeper/goalpost, etc. 
Use the position of the goalkeeper and the ball to estimate distance between goalpost and ball.
Dont refer to detection, frame numbers.
No need to say redundant information given in the data.
Be precise and accurate to match the ground truth annotations.
Don't include your incapability to answer. Be confident in your answers and don't include uncertainty in your language.
Dont list entities instead try to answer in rich linguistic and descriptive paragraphs without headings.
Compile to a long paragraph like in news articles.
''')



csv_filename = "/home/sushant/D1/DataSets/SoccerNet-tracking/tracking-2023/test/SNMOT-117/gt/gt.txt"
csv_data = {}
sampling_rate = 30

with open(csv_filename, "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        if int(row[0]) % sampling_rate == 0:
            key = filtered_config[int(row[1])]
            if key not in csv_data:
                csv_data[key] = []
            csv_data[key].append({int((row[0]))  :[ int(e) for e in row[2:6]]})

pixel_coordinates_to_xyxyn = lambda pixel_coords: [
    pixel_coords[0] / 1920,
    pixel_coords[1] / 1080,
    (pixel_coords[2] - pixel_coords[0]) / 1920,
    (pixel_coords[3] - pixel_coords[1]) / 1080
]

for key, rows in csv_data.items():
    print(f"\n{key}")
    for row in rows:
        for k,v in row.items():
            row[k] = pixel_coordinates_to_xyxyn(v)
            row[k] = [round(e, 2) for e in row[k]]
            frame_data = str(row)
            frame_data = frame_data.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
            print(frame_data)
