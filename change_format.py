from config import get_config
import json
import pandas as pd

config = get_config()

# Function to read sentences from a text file
def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

# Paths to your text files
eng_file_path = 'eng-ukr-dataset/eng1.txt'
ukr_file_path = 'eng-ukr-dataset/ukr1.txt'

# Read sentences from the files
eng_sentences = read_sentences(eng_file_path)
ukr_sentences = read_sentences(ukr_file_path)

# Check if the lengths match
if len(eng_sentences) != len(ukr_sentences):
    raise ValueError('Mismatched lengths between English and Ukrainian sentences.')

# Create a list of dictionaries
data_list = [{'id': i, 'translation': {'eng': eng, 'ukr': ukr}} for i, (eng, ukr) in enumerate(zip(eng_sentences, ukr_sentences))]

# Convert the list of dictionaries to a JSON string
json_data = json.dumps(data_list, ensure_ascii=False, indent=2)

# Save the JSON string to a file
json_path = 'eng-ukr-dataset/data.json'
with open(json_path, 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

# Optionally, you can also load the JSON string into a pandas DataFrame
df = pd.read_json(json_data)
print(df.head())
