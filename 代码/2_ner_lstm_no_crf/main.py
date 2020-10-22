import os
import json
import time
import argparse

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam

from dataloader import DuIE_Dataset
from metric import F1_ner
from model import LSTMTagger

import numpy as np






class Runner(object):
    def __init__(self):
        self.model_dir = 'saved_models'
        self.gpu = 0
        self.ner_metrics = F1_ner()
        
        self.model = None


    def _init_model(self):
        self.model = LSTMTagger(self.gpu).cuda(self.gpu)
        self.optimizer = Adam(self.model.parameters())

    def run(self, mode: str):
        if mode == 'train':
            self._init_model()
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=0)
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, '_' + str(epoch)))

    def evaluation(self):
        dev_set = DuIE_Dataset("dev_data.json")
        loader = DataLoader(dev_set, batch_size=2, pin_memory=True)
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                pred, gold = self.model(sample, is_train=False)
                self.ner_metrics(gold, pred)

            ner_result = self.ner_metrics.get_metric()
            print('NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ]))

    def train(self):
        train_set = DuIE_Dataset("train_data.json")
        loader = DataLoader(train_set, batch_size=256, pin_memory=True, shuffle=True)

        for epoch in range(50):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output
                loss.backward()
                self.optimizer.step()

                pbar.set_description("L: {:.4f} epoch: {}/{}:".format(
            loss.item(), epoch, 50))

            self.save_model(epoch)

            self.evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        '-m',
                        type=str,
                        default='evaluation',
                        help='train|evaluation')
    args = parser.parse_args()
    config = Runner()
    config.run(mode=args.mode)