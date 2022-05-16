import pandas as pd

food_df = pd.read_csv('cleaned_new_mach3.csv')
clean_food = pd.DataFrame()
unique_stories = set(food_df['story_id'])
# unique_stories = ["0008610c8126e6f2149fe984c0c77d0b"]
for story in unique_stories:
    story_temp_df = food_df[food_df['story_id'] == story]
    clean_rows = pd.DataFrame(columns=['story_id', 'start_char', 'end_char', 'date', 'text'])
    for row_index in range(len(story_temp_df)):
        row = story_temp_df.iloc[row_index]
        print(row_index)
        try:
            next_row = story_temp_df.iloc[row_index + 1]
            date = row['date']
            start_index_of_next = next_row['start_char']
            end_index_of_current = row['end_char']
            if start_index_of_next == (end_index_of_current + 1):
                print('here')
                start_index_of_current = row['start_char']
                end_index_of_next = row['end_char']
                food_text = row['text'] + ' ' + next_row['text']
                smushed_row = {'start_char': start_index_of_current, 'end_char': end_index_of_next,
                               'story_id': story, 'date': date, 'text': food_text}
                print(smushed_row)
                print(clean_rows)
                clean_rows = clean_rows.append(smushed_row, ignore_index=True)
                row_index += 2
            else:
                clean_rows = clean_rows.append(row, ignore_index=True)

        except Exception as e:
            print(e)
    clean_rows = clean_rows.drop_duplicates(subset=['text'], keep='first')
    clean_food = clean_food.append(clean_rows)

clean_food.to_csv('cleaned_new_mach3_2.csv')
