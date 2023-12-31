'''
Config for robust access to hyparameters, paths and specifics of the dataset
'''

from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 3,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "eng", # input language
        "lang_trgt": "ukr", # output language
        "dataset_path": "eng-ukr-dataset",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / config['model_folder'] / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(config['model_folder']).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
