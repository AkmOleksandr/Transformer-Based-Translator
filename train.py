'''
Take a batch of train data (a single sentence), take labels (same sentence of target language), compute loss, optimize with Adam, run validation (... take this,  pass through encoder, receive encoder output (computed once). Then pass encoder output( embedding of the shape... where each row is...) along with empty decoder output (<SOS> token)..., predict a single word, append it to decoder output and pass the encoder output along with update decoder output(<SOS> 'word'), until we ...) )  
'''
# From files
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_sentences(ds, lang): # get all sentences of the dataset (method will differ depending on structure of the dataset)
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang): # config -  access dictionary with hyparameters and paths, ds - dataset, lang - language

    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # each lang has diff tokenizer
    if not Path.exists(tokenizer_path): # if doesn't exist
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # tokenizer encouters uknown word - replaces it with UNK
        tokenizer.pre_tokenizer = Whitespace() # split by white spce
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # word level tokenizer, special tokens cover words that are unknown, paddings, start and end of sentence. we include word into vocabulary only if it appears 2 or more times
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # tokenize all sentences of the dataset of given language
        tokenizer.save(str(tokenizer_path)) # save it
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) # import if exists
    return tokenizer

def get_ds(config): # load, split the dataset for training and testing and tokenize it
    
    ds_raw = load_dataset('json', data_files='eng-ukr-dataset/data.json', split='train') # initial dataset

    # Build tokenizers for source and target data
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_trgt = get_or_build_tokenizer(config, ds_raw, config['lang_trgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # train and test datasets in raw text

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trgt, config['lang_src'], config['lang_trgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trgt, config['lang_src'], config['lang_trgt'], config['seq_len'])
    
    # Create DataLoader objects
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt # return train_dataloader (object that contains data to use for robust training), val_dataloader for validation and tokenizers for both source and target objects (they are models to tokenize different languages)


def get_model(config, vocab_src_len, vocab_trgt_len): # build the model
    model = build_transformer(vocab_src_len, vocab_trgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model


def train_model(config):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    
    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt = get_ds(config) # get data and tokenizers
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trgt.get_vocab_size()).to(device) # get model

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9) # initialize optimizer

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # CrossEntropy Loss

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()    
        model.train() # every epoch go back to training mode after the validation (could be done with metrics)
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}") # splits data into batches and visualizes process of training

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device) 
            encoder_mask = batch['encoder_mask'].to(device) 
            decoder_mask = batch['decoder_mask'].to(device)

            # Pass batch of training data through encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) 

            label = batch['label'].to(device) # extract labels from the batch

            # Compute the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_trgt.get_vocab_size()), label.view(-1)) # output and label as inputs into the loss

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) # print loss along with the progress bar

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation after each epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step) # batch_iterator for visualization of the bar and loss

        # Save the model after each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        print("Saving model to:", model_filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_trgt, max_len, device, print_msg, global_step, num_examples=2):

    model.eval()
    count = 0

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # if we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds: # for every batch
            count += 1
            encoder_input = batch["encoder_input"].to(device) # encoder input for a batch=1, 1 tokenied sentence
            encoder_mask = batch["encoder_mask"].to(device) # mask fir the encoder

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trgt, max_len, device) # prediction of the sentence in tokens

            source_text = batch["src_text"][0] # original source sentence
            target_text = batch["trgt_text"][0] # original target sentence
            model_out_text = tokenizer_trgt.decode(model_out.detach().cpu().numpy()) # prediction of the model about target sentence
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples: # how many examples to show during validation (*by eye*, metrics could be added)
                print_msg('-'*console_width)
                break

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trgt, max_len, device): # loads model, encoder_input, encoder_mask, tokenizer of source lang, tokenizer of trgt lang, max_len of sentence, device

    sos_idx = tokenizer_trgt.token_to_id('[SOS]') # create <SOS> token
    eos_idx = tokenizer_trgt.token_to_id('[EOS]') # create <EOS> token

    encoder_output = model.encode(encoder_input, encoder_mask) # compute encoder output once

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device) # initialize the decoder input with <SOS>

    while decoder_input.size(1) < max_len: # until decoder_input size is less than max_len(seq_len)

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
    
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # reuse 

       
        probs = model.project(out[:, -1]) # probabilities of the next token
        
        _, next_word = torch.max(probs, dim=1) # get the token with max prob

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1) # append next_word (the predicted word) to decoder_input

        if next_word == eos_idx: # if next token is <EOS> break
            break

    return decoder_input.squeeze(0) # return decoder (prediction of the whole sentence)
 
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
