import torch
import numpy as np
#1.定义超参数
vocab = {}
vocab_size = 7
num_inputs, num_hiddens, num_outputs = 2, 2, 4


def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), dtype=torch.float)
        return torch.nn.Parameter(ts, requires_grad=True)

def _three():
    return (_one((num_inputs, num_hiddens)), 
            _one((num_hiddens, num_hiddens)),
            torch.nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)))
def LSTM(inputs, H_0):
    #1.初始化模型参数
    w_xi, w_hi,b_i = _three()
    w_xf, w_hf,b_f = _three()
    w_xo, w_ho,b_o = _three()
    w_xc, w_hc,b_c = _three()
    
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    # 2.定义模型
    H,C= H_0, H_0
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, w_xi) + torch.matmul(H,w_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, w_xf) + torch.matmul(H,w_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, w_xo) + torch.matmul(H,w_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, w_xc) + torch.matmul(H,w_hc) + b_c)

        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,C)
    

tokens_id = torch.tensor([0,1,2,3,4,5,5,5,6])
embeds = torch.nn.Embedding(vocab_size,num_inputs)
current_word_embedding = embeds(tokens_id)
#1.current_word_embedding（句子的长度,词嵌入维度)
H_0 = torch.zeros(num_hiddens) 
O, last_H = RNN(current_word_embedding, H_0)


torch_rnn = torch.nn.LSTM(num_inputs, num_hiddens, bidirectional=False, batch_first=True)
O_2, last_H_2 = torch_rnn(current_word_embedding.unsqueeze(0)) 
print('done')

