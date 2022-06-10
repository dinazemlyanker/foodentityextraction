import pickle
import spacy
import en_core_web_lg
from thinc.api import Config
from spacy import Language
from spacy.lang.en import English
import sys
import re
import random

with open("spacy_model.p", "rb") as h:
	take = pickle.load(h)
config = Config().from_disk("./config.cfg")
lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
nlp = lang_cls.from_config(config)
nlp = nlp.from_bytes(take)



def make_word_dict(take_1):
	output = dict()
	for thing in take_1[-1]['entities']:
 		if thing[-1] == "FOOD":
 			corners = [thing[0], thing[1]]
 			output[take_1[0][corners[0]:corners[1]]] = thing
	return output


def has_space(string, i1, i2):
	while i1 <= i2:
		if string[i1:i1+1] == " ": return True
		i1++1
	return false


def check_for_multis(take_1, word_dict):
	sentence = take_1[0]
	doc = nlp(sentence)
	for chunk in doc.noun_chunks:
		if len(chunk) > 1:
			#print(chunk.text)
			cat_tokens = []
			for token in chunk:
				if word_dict.get(token.text, -1) != -1:
					cat_tokens.append(word_dict[token.text])
			if len(cat_tokens) > 1 and len(cat_tokens) < 4:
				new_entry = [cat_tokens[0][0], cat_tokens[-1][1], 'FOOD']
				print(sentence[new_entry[0]:new_entry[1]])
				take_1[-1]['entities'][0] = new_entry
	return take_1[-1]['entities']

# take1 = pickle.load(open( "parrot2.pkl", "rb" ))
# take2 = pickle.load(open( "parrot3.pkl", "rb" ))
# print(len(take1), len(take2))
# print(take1[0])
# print(take2[0])
# print(take1[-1])
# print(take2[-1])

with open("parrot2.pkl", "rb") as h:
	take = pickle.load(h)
	for x in range(len(take)):
		# 	[-1]['entities'])
		# print(take[x][0])
		word_dict = make_word_dict(take[x])
		#print(word_dict)
		take[x][-1]['entities'] = check_for_multis(take[x], word_dict)
		word_dict = make_word_dict(take[x])
		print(word_dict)
		# print(take[x][-1]['entities'])
with open('parrot22.pkl', 'wb') as f:
     pickle.dump(take, f)

# # config = Config().from_disk("./config.cfg")
# # lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
# # nlp = lang_cls.from_config(config)
# # nlp.from_bytes(take)

# # doc = nlp(sys.argv[1])
# # for token in doc:
# # 	print(token.text, token.pos_)
# # for chunk in doc.noun_chunks:
# # 	print(chunk.text, chunk.root.text)
# # for ents in doc.ents:
# # 	print(ents.text, ents.label_)



# #ALGORITHM:
# #for every chunk in a doc, check if several annotations are present. if so, merge together.
# #dot his throughout all of pkl
# #then send it to the data and start machine
