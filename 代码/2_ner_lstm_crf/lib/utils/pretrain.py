import torch 
import torch.nn as nn
import os
import numpy as np 
from collections import OrderedDict
import json

class PretranEmbedding(object):
    def __init__(self, hyper, is_first_time=False):
        self.hyper = hyper
        self.embed_file = os.path.join(self.hyper.data_root, 'w2v.txt')
        self.stoi = OrderedDict()
        self.itos = OrderedDict()
        if is_first_time:
            self.word_vocab = json.load(open(os.path.join(self.hyper.data_root, 'word_vocab.json')))
            if len(self.word_vocab) > 0:
                self.embed_file = self.gen_new_w2v()
            else:
                print('please generate word_vocab first')
        self.vectors = self.get_vectors(self.embed_file)
       
    def gen_new_w2v(self):
        with open(self.embed_file, 'r') as f, open('w2v.txt', 'w') as r:
            lines = f.readlines()
            dim = 300
            word_idx = 0
            for line in lines[1:]:
                parts = line.split()
                if len(parts) != dim + 1:
                    continue
                word = parts[0]
                if word in self.word_vocab:
                    r.write(line)
        return 'w2v.txt'

    def get_vectors(self, embed_file):
        embed_matrix = list()
        with open(embed_file, 'r') as f:
            lines = f.readlines()
            dim = 300
            word_idx = 0
            for line in lines[1:]:
                parts = line.split()
                if len(parts) != dim + 1:
                    continue
                word = parts[0]
                vec = [np.float32(val) for val in parts[1:]]
                embed_matrix.append(vec)
                self.stoi[word] = word_idx
                self.itos[word_idx] = word
                word_idx += 1
        return embed_matrix



def load_pretrained_embedding(words, pretrained_vocab):
    embed = torch.zeros(len(words), len(pretrained_vocab.vectors[0]))
    dim =  len(pretrained_vocab.vectors[0])
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = torch.tensor(pretrained_vocab.vectors[idx], dtype=torch.float32)
        except KeyError:
            oov_count += 0
            if word == '<pad>':
                embed[i, :] = torch.from_numpy(np.zeros(dim, dtype=np.float32))
            else:
                embed[i, :] = torch.from_numpy(np.random.uniform(-0.25, 0.25, dim)).float()

    if oov_count > 0:
        print("There are %d oov words."%(oov_count))
    return embed