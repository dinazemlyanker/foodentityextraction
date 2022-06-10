import en_core_web_lg
import pandas as pd
import spacy
import pickle
from thinc.api import Config
import os
import json
import sys



config = Config().from_disk("./config.cfg")
lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
nlp = lang_cls.from_config(config)
with open("nlp.p", "rb") as f:
	nlp = pickle.load(f)


story_dir = "storyjsons"
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
					print(ent.text)
	except:
		print("HEEEEEE")