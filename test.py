from datasets import load_dataset
from datasets import IterableDataset
from config import get_config

dataset = load_dataset('csv', data_files='eng-ukr-dataset/data.csv')
print(dataset)