import pickle

with open('nlp.p', 'rb') as f:
    nlp = pickle.load(f)

doc = nlp('I am baking a cake.')
for ent in doc.ents:
    print(ent)