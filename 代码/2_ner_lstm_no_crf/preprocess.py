import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property


class Preprocessing(object):
    def __init__(self, path):
        self.raw_data_root = os.path.join(path, 'raw_data')
        self.data_root = os.path.join(path, 'data')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)


    def gen_bio_vocab(self):
        result = {'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))


    def gen_vocab(self, min_freq: int):
        source = os.path.join(self.raw_data_root, "train_data.json")
        target = os.path.join(self.data_root, 'word_vocab.json')

        cnt = Counter()  # 8180 total
        with open(source, 'r') as s:
            for line in s:
                line = line.strip("\n")
                if not line:
                    return None
                instance = json.loads(line)
                text = list(instance['text'])
                cnt.update(text)
        result = {'<pad>': 0}
        i = 1
        for k, v in cnt.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result['oov'] = i
        json.dump(result, open(target, 'w'), ensure_ascii=False)

    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text']

        bio = None

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            if not self._check_valid(text):
                return None

            entities: List[str] = self.spo_to_entities(text, spo_list)

            bio, entities_list = self.spo_to_bio(text, entities)

        result = {
            'text': text,
            'bio': bio,
            'entities_list':entities_list
        }
        return json.dumps(result, ensure_ascii=False)

    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write('\n')

    def gen_all_data(self):
        self._gen_one_data("train_data.json")
        self._gen_one_data("dev_data.json")

    def _check_valid(self, text: str) -> bool:
        if len(text) > 150:
            return False
        return True

    def spo_to_entities(self, text: str,
                        spo_list: List[Dict[str, str]]) -> List[str]:
        entities = set(t['object'] for t in spo_list) | set(t['subject']
                                                            for t in spo_list)
        return list(entities)


    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        bio = ['O'] * len(text)
        entities_list = []
        for e in entities:
            begin = text.find(e)
            end = begin + len(e) - 1

            assert end <= len(text)

            entities_list.append({
                'text' : e,
                'begin':begin,
                'end':end
            })

            bio[begin] = 'B'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        return bio, entities_list


if __name__ == "__main__":
    preprocess = Preprocessing(os.getcwd())
    preprocess.gen_vocab(1)
    preprocess.gen_bio_vocab()
    preprocess.gen_all_data()

