import pandas as pd

# display text, have an entry box for the food entities, process the input by line
# extract the entities with NER model, process the lists according to this to get the scores
# and any false negatives or positives, displays the information and saves it to a csv?
extracted_df = pd.read_csv('cleaned_new_mach3.csv')
true_df = pd.read_csv('manual.csv')
print(true_df.columns)
print(extracted_df.columns)

story_result_list = []
for _, row in true_df.iterrows():
    # print('here')
    # print(row)
    story_id = row['story_id']
    extracted_food_entities = list((extracted_df[extracted_df["story_id"] == story_id])['text'])
    true_food_entities = row['food_entities']
    true_food_entities = list(map(str.strip, true_food_entities.split(',')))


    tp = 0
    fp = 0
    fn = 0
    false_neg_list = []
    false_pos_list = []

    for food in true_food_entities:
        if food in extracted_food_entities:
            tp += 1
        else:
            fn += 1
            false_neg_list.append(food)
    for food in extracted_food_entities:
        if food not in true_food_entities:
            false_pos_list.append(food)
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    story_result_dict = {'story_id': story_id, 'precision': precision, 'recall': recall, 'f1': f1,
                         'false_negs': false_neg_list, 'false_pos': false_pos_list}
    story_result_list.append(story_result_dict)

result_df = pd.DataFrame(story_result_list)
result_df.to_csv('tester_results.csv')
