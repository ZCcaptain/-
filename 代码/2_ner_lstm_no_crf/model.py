import torch
import torch.nn as nn
import torch.nn.functional as F
from CRF import CRF
from torch.nn import Parameter
import os
import json

class LSTMTagger(nn.Module):

    def __init__(self, gpu=0):
        super(LSTMTagger, self).__init__()
        self.gpu = gpu
        self.data_root = os.path.join(os.getcwd(), 'data')
        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        self.id2bio = {i:t for t, i in self.bio_vocab.items()}
        self.embedding_dim, self.hidden_dim, self.vocab_size, self.tagset_size = 200, 300, len(self.word_vocab), len(self.bio_vocab)

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim,
                                self.hidden_dim,
                                bidirectional=True,
                                batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        # self.tagger = CRF(self.tagset_size, batch_first=True)
    def forward(self, samples, is_train=False):
        tokens_id = samples[0].cuda(self.gpu)
        bio_id = samples[1].cuda(self.gpu)
        length = samples[2]
        text = samples[3]
        bio = samples[4]

        embeds = self.word_embeddings(tokens_id)

        lstm_out, _ = self.lstm(embeds)

        lstm_out = (lambda a: sum(a) / 2)(torch.split(lstm_out, self.hidden_dim, dim=2))

        tag_scores = self.hidden2tag(lstm_out)

        if is_train:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(tag_scores.view(-1, self.tagset_size), bio_id.view(-1))
            return loss
        else:
            pred = tag_scores.data.max(-1, keepdim=True)[1].squeeze(-1)
            pred_tag = [[ self.id2bio[tag] for tag in batch.cpu().numpy().tolist() ] for batch in pred]
            return pred_tag, [tag.split() for tag in bio]
