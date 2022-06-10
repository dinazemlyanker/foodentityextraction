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
import sys
from thinc.api import Config
import wget
import math


#This function takes the storyid for a storyjson blog post, and returns a dataframe with expected and actual results for each term in the post.
def build_food_tester(convert):

	#First, do the csv results.
	jsons_export = dict()
	filepath = "./storyjsons/" + convert + ".json"
	with open(filepath, 'rb') as json_file:
		FUCK = json.load(json_file)
		HELP_MEEEEE = FUCK['content']
		doc = nlp(HELP_MEEEEE) #Don't worry about any of these :)

		for ents in doc.ents:
			if(ents.label_ != "FOOD"): continue
			elif jsons_export.get(ents.text, -1) != -1:
				jsons_export[ents.text] += 1
			else:
				jsons_export[ents.text] = 1
	
	#Next, collect the data from the Manual Tests
	filepath = "./csvs/" + convert + ".csv"
	csvs_export = pd.read_csv(filepath)
	csvs_export = dict(zip(csvs_export.iloc[:,0], csvs_export.iloc[:,1]))
	#print(csvs_export)

	#Finally, put the lists together.
	df = pd.DataFrame()
	df = df.reindex(columns = ["term", "expected", "actual"])
	for key in csvs_export:
		value3 = 0
		if jsons_export.get(key, -1) != -1:
			value3 = jsons_export[key]
			jsons_export[key] = -2
		df.loc[len(df.index)] = [key, csvs_export[key], value3]

	for key in jsons_export:
		if jsons_export[key] != -2:
			df.loc[len(df.index)] = [key, 0, jsons_export[key]]
	df = df.sort_values('term')
	return df


#This gives precision and recall numbers for a given dataframe created above.
def precall(data):
	false_positives = 0
	false_negatives = 0
	true_positives = 0
	for x in range(len(data)):
		expected = data.iloc[x,1]
		actual = data.iloc[x,2]
		if math.isnan(expected - actual): continue
		else:
			if expected == actual:
				true_positives += actual
			elif expected > actual:
				true_positives += actual
				false_negatives += (expected - actual)
			else:
				true_positives += expected
				false_positives += (actual - expected)
	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)
	print("PRECISION: " + str(precision))
	print("RECALL: " + str(recall))
	return [precision, recall]


#Opening model...
with open("spacy_model.p", "rb") as h:
	take = pickle.load(h)
config = Config().from_disk("./config.cfg")
lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
nlp = lang_cls.from_config(config)
nlp = nlp.from_bytes(take)


#j_to_c exists to ensure that everything in the csv folder of our answers is in the storyjson on its own.
#This also means that I haven't created tests yet for Instagram captions and the magazine covers.
j_to_c = dict()
contain = os.listdir("storyjsons")
for index in range(len(contain)):
	j_to_c[contain[index][0:-5]] = index


pd_df = pd.DataFrame(columns = ["story", "precision", "recall"])
item_list = os.listdir("csvs")
#This will run the main system for each storyjson file, and create a comparison table each time.
for element in item_list:	
	convert = element[0:-4]
	if j_to_c.get(convert) is not None:
		df = build_food_tester(convert)
		print(df)
		precall1 = precall(df)
		precall1[0] *= 100
		precall1[1] *= 100
		pd_df.loc[len(df.index)] = [convert, precall1[0], precall1[1]]
print("AVERAGE PRECISION: " + str(pd_df["precision"].mean()))
print("AVERAGE RECALL: " + str(pd_df["recall"].mean()))
pd_df.to_csv('06-10-2022_manual_testing_modelv1.csv')



##Legacy Code
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# # with open("nlp1.pkl", "rb") as h:
# # 	take = pickle.load(h)
# url = "https://mediacloud-ihop.s3.amazonaws.com/models/spacy_model.p"
# take1 = wget.download(url)

# for token in doc:
# 	print(token.text, token.pos_)
# for chunk in doc.noun_chunks:
# 	print(chunk.text, chunk.root.text)
# for ents in doc.ents:
# 	print(ents.text, ents.label_)
