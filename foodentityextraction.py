# download spacy language model

# import libraries
import json
import os
from json import JSONDecodeError

import en_core_web_lg
import pandas as pd
import re
import random
import spacy
from spacy import Language
from spacy.util import minibatch, compounding
import warnings
from spacy.training import Example
import pickle
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np


# nlp = Language.from_config(config)
# read in the food csv file
if False:
    food_df = pd.read_csv("food.csv")

    # print row and column information
    food_df.head()

    # disqualify foods with special characters, lowercase and extract results from "description" column
    foods = food_df[food_df["description"].str.contains("[^a-zA-Z ]") == False]["description"].apply(lambda food: food.lower())

    # filter out foods with more than 3 words, drop any duplicates
    foods = foods[foods.str.split().apply(len) <= 3].drop_duplicates()

    # find one-worded, two-worded and three-worded foods
    one_worded_foods = foods[foods.str.split().apply(len) == 1]
    two_worded_foods = foods[foods.str.split().apply(len) == 2]
    three_worded_foods = foods[foods.str.split().apply(len) == 3]

    # total number of foods
    total_num_foods = round(one_worded_foods.size / 45 * 100)

    # shuffle the 2-worded and 3-worded foods since we'll be slicing them
    two_worded_foods = two_worded_foods.sample(frac=1)
    three_worded_foods = three_worded_foods.sample(frac=1)

    # append the foods together
    foods = one_worded_foods.append(
        two_worded_foods[:round(total_num_foods * 0.30)]).append(three_worded_foods[:round(total_num_foods * 0.25)])

    food_templates = [
        "I ate my {}",
        "I'm eating a {}",
        "I just ate a {}",
        "I only ate the {}",
        "I'm done eating a {}",
        "I've already eaten a {}",
        "I just finished my {}",
        "When I was having lunch I ate a {}",
        "I had a {} and a {} today",
        "I ate a {} and a {} for lunch",
        "I made a {} and {} for lunch",
        "I ate {} and {}",
        "today I ate a {} and a {} for lunch",
        "I had {} with my husband last night",
        "I brought you some {} on my birthday",
        "I made {} for yesterday's dinner",
        "last night, a {} was sent to me with {}",
        "I had {} yesterday and I'd like to eat it anyway",
        "I ate a couple of {} last night",
        "I had some {} at dinner last night",
        "Last night, I ordered some {}",
        "I made a {} last night",
        "I had a bowl of {} with {} and I wanted to go to the mall today",
        "I brought a basket of {} for breakfast this morning",
        "I had a bowl of {}",
        "I ate a {} with {} in the morning",
        "I made a bowl of {} for my breakfast",
        "There's {} for breakfast in the bowl this morning",
        "This morning, I made a bowl of {}",
        "I decided to have some {} as a little bonus",
        "I decided to enjoy some {}",
        "I've decided to have some {} for dessert",
        "I had a {}, a {} and {} at home",
        "I took a {}, {} and {} on the weekend",
        "I ate a {} with {} and {} just now",
        "Last night, I ate an {} with {} and {}",
        "I tasted some {}, {} and {} at the office",
        "There's a basket of {}, {} and {} that I consumed",
        "I devoured a {}, {} and {}",
        "I've already had a bag of {}, {} and {} from the fridge",
        "I put the {} on the stove in a pan",
        "Put 1/3 cup of {}, 1/2 cup of {} and 1 cup of {} into the bowl",
        "{} is known for its health benefits like lowering blood pressure",
        "At lunch, I had some of the best {} and best {} at a restaurant in Buenos Aires",
        "At medium speed, use the electric mixer to beat {} with {} and {}",
        "For Christmas eve dinner, the family loved eating {} and {}",
        "{} skins are charred in a fryer on broil",
        "{} sometimes contains {}, {} and even {}",
    ]

    # create dictionaries to store the generated food combinations. Do note that one_food != one_worded_food. one_food == "barbecue sauce", one_worded_food == "sauce"
    TRAIN_FOOD_DATA = {
        "one_food": [],
        "two_foods": [],
        "three_foods": []
    }

    TEST_FOOD_DATA = {
        "one_food": [],
        "two_foods": [],
        "three_foods": []
    }

    # one_food, two_food, and three_food combinations will be limited to 167 sentences
    FOOD_SENTENCE_LIMIT = 167

    # helper function for deciding what dictionary and subsequent array to append the food sentence on to
    def get_food_data(count):
        return {
            1: TRAIN_FOOD_DATA["one_food"] if len(TRAIN_FOOD_DATA["one_food"]) < FOOD_SENTENCE_LIMIT else TEST_FOOD_DATA["one_food"],
            2: TRAIN_FOOD_DATA["two_foods"] if len(TRAIN_FOOD_DATA["two_foods"]) < FOOD_SENTENCE_LIMIT else TEST_FOOD_DATA["two_foods"],
            3: TRAIN_FOOD_DATA["three_foods"] if len(TRAIN_FOOD_DATA["three_foods"]) < FOOD_SENTENCE_LIMIT else TEST_FOOD_DATA["three_foods"],
        }[count]

    # the pattern to replace from the template sentences
    pattern_to_replace = "{}"

    # shuffle the data before starting
    foods = foods.sample(frac=1)

    # the count that helps us decide when to break from the for loop
    food_entity_count = foods.size - 1

    # start the while loop, ensure we don't get an index out of bounds error
    while food_entity_count >= 2:
        entities = []

        # pick a random food template
        sentence = food_templates[random.randint(0, len(food_templates) - 1)]

        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall(pattern_to_replace, sentence)

        # for each brace, replace with a food entity from the shuffled food data
        for match in matches:
            food = foods.iloc[food_entity_count]
            food_entity_count -= 1

            # replace the pattern, but then find the match of the food entity we just inserted
            sentence = sentence.replace(match, food, 1)
            match_span = re.search(food, sentence).span()

            # use that match to find the index positions of the food entity in the sentence, append
            entities.append((match_span[0], match_span[1], "FOOD"))

    # append the sentence and the position of the entities to the correct dictionary and array
    get_food_data(len(matches)).append((sentence, {"entities": entities}))



# combine the food training data
# TRAIN_FOOD_DATA_COMBINED = TRAIN_FOOD_DATA["one_food"] + TRAIN_FOOD_DATA["two_foods"] + TRAIN_FOOD_DATA["three_foods"]

nlp = en_core_web_lg.load()
# foodner_train_df = pd.read_csv('foodner_train_df_full.csv')
# for _, row in foodner_train_df.iterrows():
#     train_data_tup = (row['sent_text'], {'entities': row['entities']})
#     TRAIN_FOOD_DATA_COMBINED.append(train_data_tup)
with open('revision_data.pkl', 'rb') as f:
    TRAIN_REVISION_DATA = pickle.load(f)

if False:
    with open('food_entity_list_pickled.pkl', 'rb') as f:
        TRAIN_FOOD_DATA_COMBINED = pickle.load(f)

with open('food_entity_list_dict_pickled.pkl', 'rb') as f:
      train_food_dict = pickle.load(f)


# print(TRAIN_FOOD_DATA_COMBINED[0])
# print(TRAIN_FOOD_DATA_COMBINED[5])
# print('correct data')
# print(TRAIN_REVISION_DATA[0])
#
# # print the length of the food training data
# print("FOOD", len(TRAIN_FOOD_DATA_COMBINED))

TRAIN_FOOD_DATA_COMBINED = []
print(train_food_dict[0][0])
for key, training_list in train_food_dict.items():
    try:
        current_train_list = random.sample(training_list, 600)
    except ValueError:
        current_train_list = training_list
    for ex in current_train_list:
        TRAIN_FOOD_DATA_COMBINED.append(ex)
    print(key)
    print(len(current_train_list))
    # print(current_train_list)
    # TRAIN_FOOD_DATA_COMBINED.append(current_train_list)

# TRAIN_FOOD_DATA_COMBINED = random.sample(TRAIN_FOOD_DATA_COMBINED, 6000)
# TRAIN_FOOD_DATA_COMBINED = list(np.array(TRAIN_FOOD_DATA_COMBINED).flatten())
print(type(TRAIN_FOOD_DATA_COMBINED))
print(type(TRAIN_FOOD_DATA_COMBINED[0]))
print(TRAIN_FOOD_DATA_COMBINED[0])
# for tup in TRAIN_FOOD_DATA_COMBINED:
#     food_tup_list = tup[1]['entities']
#     sent = tup[0]
#     for food_tup in food_tup_list:
#         start_index = food_tup[0]
#         end_index = food_tup[1]
#         print(sent[start_index:end_index])

# print the length of the revision training data
print("REVISION", len(TRAIN_REVISION_DATA))
print(type(TRAIN_REVISION_DATA))
print(type(TRAIN_REVISION_DATA[0]))
print(TRAIN_REVISION_DATA[0])

# join and print the combined length
TRAIN_DATA = TRAIN_REVISION_DATA + TRAIN_FOOD_DATA_COMBINED
print("COMBINED", len(TRAIN_DATA))


# add NER to the pipeline and the new label
ner = nlp.get_pipe("ner")
ner.add_label("FOOD")

# get the names of the components we want to disable during training
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec", "lemmatizer"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
print(other_pipes)

# start the training loop, only training NER
epochs = 15
optimizer = nlp.resume_training()
with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
    warnings.filterwarnings("once", category=UserWarning, module='spacy')
    sizes = compounding(1.0, 4.0, 1.001)

    # batch up the examples using spaCy's minibatc
    for epoch in range(epochs):
        examples = TRAIN_DATA
        random.shuffle(examples)
        batches = minibatch(examples, size=sizes)
        losses = {}
        for batch in batches:
            try:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)
            except Exception as er:
                print(er)

        print("Losses ({}/{})".format(epoch + 1, epochs), losses)

doc1 = (nlp("I had a hamburger and chips for lunch today."))
doc2 = (nlp("I decided to have chocolate ice cream as a little treat for myself."))
for token in doc2:
    print(token.text, token.pos_, token.dep_)
for ent in doc1.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
# nlp.to_disk("./models/foodentity")
pickle.dump(nlp, open( "nlp.p", "wb" ))

story_dir = "sitemap/storyjsons"
food_ent_list = []
for story_file_name in os.listdir(story_dir):
    story_file = os.path.join(story_dir, story_file_name)
    print(story_file)
    try:
        with open(story_file) as json_file:
            story_file = json.load(json_file)
            story_text = story_file['content']
            doc = nlp(story_text)
            story_id = story_file['id']
            date = story_file['date']
            for ent in doc.ents:
                if ent.label_ == 'FOOD':
                    food_ent_dict = {'story_id': story_id, 'date': date,
                                     'text': ent.text, 'start_char': ent.start_char, 'end_char': ent.end_char}
                    food_ent_list.append(food_ent_dict)
    except(JSONDecodeError):
        print('error retrieving json')
food_ent_df = pd.DataFrame(food_ent_list)
food_ent_df.to_csv('food_entity_df_stratified_mach2.csv')

