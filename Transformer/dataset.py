'''
Downloading the dataset and returning ... for ... stages in the process (what input encoder what's input in decoder etc.)
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_trgt, src_lang, trgt_lang, seq_len): # original dataset, tokenizer of source lang, tokenizer of target lang, name of source lang, name of target lang, max length of a sentence (seq_len)
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trgt = tokenizer_trgt
        self.src_lang = src_lang
        self.trgt_lang = trgt_lang

        # create tokens for special words
        self.sos_token = torch.tensor([tokenizer_trgt.token_to_id("[SOS]")], dtype=torch.int64) 
        self.eos_token = torch.tensor([tokenizer_trgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_trgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # get item from dataset with index idx
        src_target_pair = self.ds[idx] # extract original pair
        src_text = src_target_pair['translation'][self.src_lang] # source text
        trgt_text = src_target_pair['translation'][self.trgt_lang] # target text

        # Transform the text into tokens (list of numbers representing the sentence)
        src_tokenized = self.tokenizer_src.encode(src_text).ids # for source lang
        trgt_tokenized = self.tokenizer_trgt.encode(trgt_text).ids # for target lang

        # How much padding we need to add to reach seq_len for specific target and and source sentence with index idx  
        src_num_padding_tokens = self.seq_len - len(src_tokenized) - 2  # -2 because <SOS>,<EOS> (Computed once and goes into encoder)
        trgt_num_padding_tokens = self.seq_len - len(trgt_tokenized) - 1 # -1 because <SOS> ...

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if src_num_padding_tokens < 0 or trgt_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Concat <SOS>, Tokens of src, <EOS>, padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_tokenized, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * src_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Concat <SOS>, Tokens of trgt, padding
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(trgt_tokenized, dtype=torch.int64),
                torch.tensor([self.pad_token] * trgt_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Concat Tokens of trgt, <EOS>, padding - to create label (expected output of the decoder)
        label = torch.cat(
            [
                torch.tensor(trgt_tokenized, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * trgt_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len) specific tokenied sentence of source lang
            "decoder_input": decoder_input,  # (seq_len) specific tokenied sentence of target lang without <EOS>
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) it's done to remove padding from the attention step as there's no point in keeping it
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # do the same thing but also hide next words in masked multi-head self-attention by applying causal mask 
            "label": label,  # (seq_len) specific tokenied sentence of target lang without <SOS>
            "src_text": src_text, # original source sentence
            "trgt_text": trgt_text, # original target sentence
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int) #...
    return mask == 0 