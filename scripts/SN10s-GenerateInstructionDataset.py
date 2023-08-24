'''
# Generate the mapping between the index and the video file name
import os
import json

source_folder = "./SN10s-min5c_collapsed_ln"
files = os.listdir(source_folder)

index_mapping = {}
for idx, file_name in enumerate(files):
    if file_name.endswith('.mp4'):
        file = file_name.rsplit('.', 1)[0]
        index_mapping[idx] = file

with open('SN10s-min5c_collapsed_ln_map.json', 'w') as json_file:
    json.dump(index_mapping, json_file, indent=4)

'''
import random
random.seed(42)

all_events = ['Goal', 'Shots', 'Foul', 'Ball out of play', 'Throw-in']

positive_question_templates = [
    "Was this a fantastic {event} or not?",
    "Did the players just execute an impressive {event}?",
    "Can we confirm that this was a successful {event}?",
    "Was the crowd thrilled by the {event} that just happened?"
]

positive_answer_templates = [
    "Absolutely, that was indeed a {event}!",
    "Yes, the players executed a perfect {event}!",
    "Definitely, that was a successful {event}!",
    "Absolutely, the {event} just took place!"
]

negative_question_templates = [
    "Was this a {neg_event} instead of {event}?",
    "Did the players intend to perform a {neg_event} here?",
    "Can we confirm that this wasn't a {event}, but rather a {neg_event}?",
    "Was there a mix-up, and it's actually a {neg_event}?"
]

negative_answer_templates = [
    "You're right, this wasn't a {event}, it was a {neg_event}.",
    "Correct, the players didn't execute a {event}, it's a {neg_event}.",
    "Exactly, this is a case of mistaken identity, it's a {neg_event}.",
    "Indeed, it's not a {event}, but a {neg_event}."
]

preample_tenplates = [
    "Deeply analyze the given short soccer game video clip which shows a particular football game event. "
    "Conduct a detailed analysis of the provided brief soccer match footage highlighting a specific game event. ",
    "Examine closely the given short football game video clip depicting a particular match incident. ",
    "Perform an in-depth analysis of the provided snippet from a soccer game, focusing on a specific event. ",
    "Thoroughly dissect the provided short video segment of a football match, showcasing a particular game event. ",
    "Carry out a comprehensive analysis of the provided soccer game clip, spotlighting a specific event within the match. ",
    "In-depth examination of the given short football game video clip, emphasizing a specific event in the game. "
]


json_question_template = "No pre-amble and always output answer in JSON with single key 'pred_evnt' and value with the most representative event shown in the video with format: { 'pred_evnt': < only one of 'Goal', 'Shots', 'Foul', 'Ball out of play', 'Throw-in' > }. "

def qa_generator(event, n_pos=3, n_neg=3):
    neg_event = random.choice(list(set(all_events) - set([event])))
    qa_set = []
    
    # Generate positive questions and answers
    for _ in range(n_pos):
        preamble = random.choice(preample_tenplates) if random.random() < 0.3 else ""
        question_template = random.choice(positive_question_templates)
        answer_template = random.choice(positive_answer_templates)
        question = question_template.format(event=event)
        answer = answer_template.format(event=event)
        qa_set.append((preamble+ question, answer))
    
    # Generate negative questions and answers
    for _ in range(n_neg):
        preamble = random.choice(preample_tenplates) if random.random() < 0.3 else ""
        question_template = random.choice(negative_question_templates)
        answer_template = random.choice(negative_answer_templates)
        question = question_template.format(event=event, neg_event=neg_event)
        answer = answer_template.format(event=event, neg_event=neg_event)
        qa_set.append([{ "from": "human","value": question+" \n<video>"}, {"from": "gpt","value": answer}])
    
    # Generate JSON event detection questions
    json_question = random.choice(preample_tenplates) + json_question_template
    json_answer = '{"pred_evnt": "'+event+'"}'

    qa_set.append([{ "from": "human","value": json_question+ " \n<video>"}, {"from": "gpt","value": json_answer}])
    
    return qa_set



import json

with open('SN10s-min5c_collapsed_ln_map.json', 'r') as json_file:
    data = json.load(json_file)

all_qa = []
for idx, filename in data.items():
    name, ifreplay, camera, event = filename.split("|")
    qa_set = qa_generator(event, n_pos=2, n_neg=1)
    for conv in qa_set:
        all_qa.append(
            {
                "id": filename,
                "video": f"{filename}.pkl",
                "conversations": conv
            }
        )
random.shuffle(all_qa)

print(len(all_qa))
print(all_qa[:20])

# dump all_qa to json min5c_collapsed_ln_dataset_train.json
with open('min5c_collapsed_ln_dataset_train.json', 'w') as json_file:
    json.dump(all_qa, json_file, indent=1)