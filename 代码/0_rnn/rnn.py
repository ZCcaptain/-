import torch
import numpy as np
#1.定义超参数
vocab = {}
vocab_size = 7
num_inputs, num_hiddens, num_outputs = 2, 2, 4


def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), dtype=torch.float)
        return torch.nn.Parameter(ts, requires_grad=True)


def RNN(inputs, H_0):
    #1.初始化模型参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    
    # 2.定义模型
    H = H_0
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H,W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
    

tokens_id = torch.tensor([0,1,2,3,4,5,5,5,6])
embeds = torch.nn.Embedding(vocab_size,num_inputs)
current_word_embedding = embeds(tokens_id)
#1.current_word_embedding（句子的长度,词嵌入维度)
H_0 = torch.zeros(num_hiddens) 
O, last_H = RNN(current_word_embedding, H_0)

torch_rnn = torch.nn.RNN(num_inputs, num_hiddens, bidirectional=False, batch_first=True)
O_2, last_H_2 = torch_rnn(current_word_embedding.unsqueeze(0), H_0.unsqueeze(0).unsqueeze(0)) 
print('done')
























