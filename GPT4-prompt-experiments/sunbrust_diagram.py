# !pip install benepar
# ! python -m spacy download en_core_web_md
# import benepar
# benepar.download('benepar_en3')

import benepar, spacy
nlp = spacy.load('en_core_web_md')

if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def find_root_verb_and_its_dobj(tree_root):
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return tree_root.lemma_, child.lemma_
        return tree_root.lemma_, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None

def find_root_verb_and_its_dobj_in_string(s):
    doc = nlp(s)
    first_sent = list(doc.sents)[0]
    return find_root_verb_and_its_dobj(first_sent.root)

import pandas as pd
import json
import tqdm



def generate_tasks_from_file(captions_list, out_filename="verb_noun.html"):
    instructions = set([task for task in captions_list])
    print(len(instructions))

    raw_phrases = []
    for instruction in tqdm.tqdm(instructions):
        try:
            verb, noun = find_root_verb_and_its_dobj_in_string(instruction)
            raw_phrases.append({
                "verb": verb,
                "noun": noun,
                "instruction": instruction
            })
        except Exception as e:
            print(e)
            print(instruction)


    raw_phrases = pd.DataFrame(raw_phrases)
    phrases = pd.DataFrame(raw_phrases).dropna()
    phrases[["verb", "noun"]].groupby(["verb", "noun"]).size().sort_values(ascending=False)
    top_verbs = phrases[["verb"]].groupby(["verb"]).size().nlargest(20).reset_index()

    df = phrases[phrases["verb"].isin(top_verbs["verb"].tolist())]
    # df = df[~df["noun"].isin(["I", "what"])]
    # df = phrases
    # df[~df["verb"].isin(top_verbs["verb"].tolist())]["verb"] = "other"
    # df[~df["verb"].isin(top_verbs["verb"].tolist())]["noun"] = "other"
    df = df.groupby(["verb", "noun"]).size().reset_index().rename(columns={0: "count"}).sort_values(by=["count"], ascending=False)
    # df = df[df["count"] > 10]
    df = df.groupby("verb").apply(lambda x: x.sort_values("count", ascending=False).head(4)).reset_index(drop=True)
    df

    import plotly.graph_objects as go
    import plotly.express as px

    # df["blank"] = "ROOT"
    # df = phrases.groupby(["verb", "noun"]).size().sort_values(ascending=False).head(5).reset_index().rename(columns={0: "count"})

    df = df[df["count"] >= 1] #########
    fig = px.sunburst(df, path=['verb', 'noun'], values='count')
    fig.update_layout(uniformtext=dict(minsize=15,
    #  mode='hide'
     ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        font_family="Times New Roman",
    )
    fig.show()
    fig.write_html(out_filename)
    # fig.savefig("output/verb_noun.pdf")


#########
machine_generated_tasks = [
    # 'Write me a story about education.',
    # 'Talk about game',
    # 'Write me a story about politics.',
    # 'I once went to America.',
    # 'Write me a story about history.',
    # 'Write me a story about politics.',
    # 'Shoot the ball.',
]

import glob
import re


json_files= glob.glob('e2_captions_tmp/*.json', recursive=True)
for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        result = re.split(r'[,.]', data['A'])
        for r in result:
                machine_generated_tasks.append(r)
generate_tasks_from_file(machine_generated_tasks, out_filename="verb_noun.html")