import torch
import random

def embedding(old_word_list, word_to_idx):
    old_word_list = list(old_word_list)
    word_list = []
    for word in old_word_list:
        if word not in word_to_idx.keys():
            word_list.append('OOV')
        else:
            word_list.append(word)
             
    token_id = [word_to_idx[token] for token in word_list]
    one_hot_mat = torch.zeros(len(word_list), len(word_to_idx), dtype=torch.float)
    for idx, id in enumerate(token_id):
        one_hot_mat[idx][id] = 1
    return one_hot_mat, token_id 
        
def test1():
    #1.建立词表
    word_to_idx = {'好': 0, '坏': 1, '优':2, '劣':3, '中':4, "OOV":5}
    #2.建立词嵌入矩阵
    pretrained_weight = torch.tensor([[1, -3],
                                    [-2,2],
                                    [1, -3],
                                    [-2,3],
                                    [0, 0],
                                    [100, -100]], dtype=torch.float)
    #3.输入句子
    word_list = "你好坏优劣不分"
    one_hot_mat, _ = embedding(word_list, word_to_idx)
    # print(one_hot_mat)

    #4.得到句子的词嵌入矩阵
    word_embedding = torch.matmul(one_hot_mat, pretrained_weight)
    print(word_embedding)

def test2():
    #1.建立词表
    word_to_idx = {'好': 0, '坏': 1, '优':2, '劣':3, '中':4, "OOV":5}
    embeds = torch.nn.Embedding(len(word_to_idx),2)
    pretrained_weight = torch.tensor([[1, -3],
                                    [-2,2],
                                    [1, -3],
                                    [-2,3],
                                    [0, 0],
                                    [100, -100]], dtype=torch.float)
    embeds.weight.data.copy_(pretrained_weight)
    #3.输入句子
    word_list = "你好坏优劣不分"
    _, tokens_id = embedding(word_list, word_to_idx)

    word_embedding = embeds(torch.tensor(tokens_id))
    print(word_embedding)


def test3():
    embeds = torch.nn.Embedding(5000, 200)

    token_id = [random.randint(0,5000) for i in range(50)]
    print(token_id)

    word_embedding = embeds(torch.tensor(token_id))
    print(word_embedding)

if __name__ == "__main__":
    # test1()
    # test2()
    test3()