import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import json
import os
import copy
from lib.utils.pretrain import PretranEmbedding, load_pretrained_embedding
import random
import math 
from typing import Dict, List, Tuple, Set, Optional
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from lib.models.CRF import CRF
import numpy as np 

class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}
        use_pretrain_embedding = True
        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
            embedding_dim=hyper.emb_size)
        if use_pretrain_embedding:
            self.pe = PretranEmbedding(self.hyper)
            self.word_embeddings.weight.data.copy_(
            load_pretrained_embedding(self.word_vocab, self.pe))
            
        self.input_dropout = nn.Dropout(p=0.5)
        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
            embedding_dim=hyper.rel_emb_size)

        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.bio_emb_size)

        self.encoder = nn.LSTM(hyper.emb_size,
                                hyper.hidden_size,
                                2,
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.5)

        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)


        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)




    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens_len = torch.tensor(list(sample.length))

        tokens = pad_sequence(sample.tokens_id, batch_first=True, padding_value=self.word_vocab['<pad>']).to(self.gpu)
        bio_gold = pad_sequence(sample.bio_id, batch_first=True, padding_value=self.bio_vocab['O']).to(self.gpu)
        max_selection_len = tokens.size()[1]

        text_list = sample.text
        spo_gold = sample.spo_gold

        bio_text = sample.bio

        if self.hyper.cell_name in ('gru', 'lstm'):
            mask = tokens != self.word_vocab['<pad>']  # batch x seq
            bio_mask = mask
        else:
            raise ValueError('unexpected encoder name!')

        if self.hyper.cell_name in ('lstm', 'gru'):
            embedded = self.word_embeddings(tokens)
            embedded = self.input_dropout(embedded)
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, tokens_len, batch_first=True)
            o, h = self.encoder(embedded)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
            o = (lambda a: sum(a) / 2)(torch.split(o,
                                                   self.hyper.hidden_size,
                                                   dim=2))
        else:
            raise ValueError('unexpected encoder name!')
        emi = self.emission(o)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = self.tagger(emi, bio_gold,
                                    mask=bio_mask, reduction='mean')
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            output['decoded_tag'] = [list(map(lambda x : self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            temp_tag = [torch.tensor(i) for i in decoded_tag]
            # for line in temp_tag:
            #     line.extend([self.bio_vocab['<pad>']] *
            #                 (self.hyper.max_text_len - len(line)))
            bio_gold = pad_sequence(temp_tag, batch_first=True, padding_value=self.bio_vocab['O']).to(self.gpu)
            # bio_gold = torch.tensor(temp_tag).cuda(self.gpu)

        
        output['loss'] = crf_loss
        output['description'] = partial(self.description, output=output)
        return output

class LSTM_Classifier(nn.Module):
    def __init__(self, hyper):
        super(LSTM_Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(embedding_size,hidden_size,bidirectional=True,batch_first=True)
        self.classifer = nn.Linear(hidden_size, label_size)

    def forward(inputs):
        inputs_embedding = self.embedding(inputs)#(句子的长度，词嵌入维度)
        _, h = self.encoder(inputs_embedding)#(lstm隐藏层维度)
        output = self.classifer(h)
        output = nn.Softmax(output)
        return output

