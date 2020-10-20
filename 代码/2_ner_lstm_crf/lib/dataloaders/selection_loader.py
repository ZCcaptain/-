import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from pytorch_transformers import *


class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []

        # for bert only
        if self.hyper.cell_name == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        if self.hyper.cell_name == 'bert':
            text, bio, selection = self.pad_bert(text, bio, selection)
            tokens_id = torch.tensor(
                self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2tensor(text)
        bio_id = self.bio2tensor(bio)
        selection_id = self.selection2tensor(text, selection)
        label_id = self.label2tensor(text, selection)

        return tokens_id, bio_id, selection_id, label_id, len(text), spo, text, bio, selection

    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text: List[str], bio: List[str], selection: List[Dict[str, int]]) -> Tuple[List[str], List[str], Dict[str, int]]:
        # for [CLS] and [SEP]
        text = ['[CLS]'] + text + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] +
                      1, 'predicate': triplet['predicate']} for triplet in selection]
        assert len(text) <= self.hyper.max_text_len
        text = text + ['[PAD]'] * (self.hyper.max_text_len - len(text))
        return text, bio, selection

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        oov = self.word_vocab['oov']
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        # padded_list.extend([self.word_vocab['<pad>']] *
        #                    (self.hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        # padded_list.extend([self.bio_vocab['O']] *
        #                    (self.hyper.max_text_len - len(bio)))
        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros(
            (self.hyper.max_text_len, len(self.relation_vocab),
             self.hyper.max_text_len))
        if 'N' in self.relation_vocab.keys():
            NA = self.relation_vocab['N']
        elif 'NA' in self.relation_vocab.keys():
            NA = self.relation_vocab['NA']
        elif 'None' in self.relation_vocab.keys():
            NA = self.relation_vocab['None']
        result[:, NA, :] = 1
        for triplet in selection:

            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']

            result[subject, predicate, object] = 1
            if predicate != NA:
                result[subject, NA, object] = 0

        return result

    def label2tensor(self, text, selection):
        # s p o
        result = torch.zeros(len(self.relation_vocab))
        if 'N' in self.relation_vocab.keys():
            NA = self.relation_vocab['N']
        elif 'NA' in self.relation_vocab.keys():
            NA = self.relation_vocab['NA']
        elif 'None' in self.relation_vocab.keys():
            NA = self.relation_vocab['None']
        result[NA] = 1
        for triplet in selection:
            predicate = triplet['predicate']
            result[predicate] = 1
            if predicate != NA:
                result[NA] = 0
        return result


class Batch_reader(object):
    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, selection_id, spo, text, bio

        self.tokens_id = transposed_data[0]
        self.bio_id = transposed_data[1]
        self.selection_id = torch.stack(transposed_data[2], 0)
        self.label_id = torch.stack(transposed_data[3], 0)
        self.length = transposed_data[4]

        self.spo_gold = transposed_data[5]
        self.text = transposed_data[6]
        self.bio = transposed_data[7]
        self.selection = transposed_data[8]

    # def pin_memory(self):
    #     self.tokens_id = self.tokens_id.pin_memory()
    #     self.bio_id = self.bio_id.pin_memory()
    #     self.selection_id = self.selection_id.pin_memory()
    #     return self


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
