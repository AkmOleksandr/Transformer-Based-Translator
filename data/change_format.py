from config import get_config
import json
import pandas as pd

config = get_config()

def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

eng_file_path = 'eng-ukr-dataset/eng.txt'
ukr_file_path = 'eng-ukr-dataset/ukr.txt'

eng_sentences = read_sentences(eng_file_path)
ukr_sentences = read_sentences(ukr_file_path)

if len(eng_sentences) != len(ukr_sentences):
    raise ValueError('Mismatched lengths between English and Ukrainian sentences.')

data_list = [{'id': i, 'translation': {'eng': eng, 'ukr': ukr}} for i, (eng, ukr) in enumerate(zip(eng_sentences, ukr_sentences))]

json_data = json.dumps(data_list, ensure_ascii=False, indent=2)

json_path = 'eng-ukr-dataset/data.json'
with open(json_path, 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

df = pd.read_json(json_data)
print(df.head())
