# load model
import spacy as spacy
from spacy import Language

import en_core_web_lg

# nlp = Language.from_config(config)
nlp = en_core_web_lg.load()
nlp = nlp.from_disk('/models/foodentity')
# get all entities
doc1 = nlp("I really enjoy eating pasta with cheese on mondays.")

# if entity is FOOD,
for ent in doc1.ents:
    if ent.label == 'FOOD':
        print(ent.text)

