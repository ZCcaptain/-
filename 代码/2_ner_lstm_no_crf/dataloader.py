import os
import json

import torch
from torch.utils.data import Dataset

from typing import Dict, List, Tuple, Set, Optional



class DuIE_Dataset(Dataset):
    def __init__(self, dataset):
        self.data_root = os.path.join(os.getcwd(), 'data')

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.text_list = []
        self.bio_list = []
        self.max_text_len = 150

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])

    def __getitem__(self, index):
        text = self.text_list[index]
        bio = self.bio_list[index]
        tokens_id = self.text2tensor(text)
        bio_id = self.bio2tensor(bio)


        return tokens_id, bio_id, len(text), text, ' '.join(bio)

    def __len__(self):
        return len(self.text_list)


    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        oov = self.word_vocab['oov']
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        padded_list.extend([self.word_vocab['<pad>']] *
                           (self.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] *
                           (self.max_text_len - len(bio)))
        return torch.tensor(padded_list)


