import json
import os
import pickle
from json import JSONDecodeError

import pandas as pd
import en_core_web_lg

food_ent_list = []
story_dir = "sitemap/storyjsons"
food_words_df = pd.read_csv('clean_food2.csv')
food_word_list = list(set(list(food_words_df['food'])))
food_word_list = [word for word in food_word_list if ' ' not in word]


nlp = en_core_web_lg.load()
for story_file_name in os.listdir(story_dir):
    story_file = os.path.join(story_dir, story_file_name)
    print(story_file)
    try:
        with open(story_file) as json_file:
            story_file = json.load(json_file)
            story_text = story_file['content']
            doc = nlp(story_text)
            story_id = story_file['id']
            for sent in doc.sents:
                sent_text = sent.text
                food_ent_dict = {'story_id': story_id,
                                 'sent_text': sent, 'entities': []}
                for tok in sent:
                    if tok.text in food_word_list:
                        tok_index_start = sent_text.index(tok.text)
                        tok_index_end = tok_index_start + len(tok.text)
                        tok_tup = (tok_index_start, tok_index_end, "FOOD")
                        food_ent_dict["entities"].append(tok_tup)
                if len(food_ent_dict["entities"]) > 0:
                    # food_ent_list.append(food_ent_dict)
                    ent_list = list(set(food_ent_dict['entities']))
                    ent_list = sorted(ent_list, key=lambda x: x[0])
                    clean_ent_list = [ent_list[0]]
                    for ent_index in range(1, len(ent_list)):
                        current_ent = ent_list[ent_index]
                        prev_ent = ent_list[ent_index - 1]
                        if current_ent[0] > prev_ent[1]:
                            clean_ent_list.append(current_ent)
                    full_tup = (sent.text, {'entities': clean_ent_list})
                    food_ent_list.append(full_tup)
    except JSONDecodeError:
        print('error retrieving json')

with open('parrot2.pkl', 'wb') as f:
     pickle.dump(food_ent_list, f)
# food_ent_df = pd.DataFrame(food_ent_list)
# food_ent_df.to_csv('foodner_train_df_full.csv')

