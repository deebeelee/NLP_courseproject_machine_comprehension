import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim

class model(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, out_dim = 2):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.ctxt_RNN = nn.LSTM(hidden_dim, hidden_dim)
        self.qna_RNN = nn.LSTM(hidden_dim, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim*2, out_dim)

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)
        
    def forward(self, context_seq, qna_seq):
        ctxt_vectors = self.embeds(torch.tensor(context_seq))
        ctxt_vectors = ctxt_vectors.unsqueeze(1)
        qna_vectors = self.embeds(torch.tensor(qna_seq))
        qna_vectors = qna_vectors.unsqueeze(1)
        _,(output1, _) = self.ctxt_RNN(ctxt_vectors)
        _,(output2, _) = self.qna_RNN(qna_vectors)
        output1 = output1.squeeze()
        output2 = output2.squeeze()
        concat = torch.cat([output1, output2], 0)
        #forward1, forward2 = hidden1[0], hidden2[0]
        #concat = torch.cat([forward, backward], 1)
        output = torch.nn.functional.relu(concat)
        prediction = self.out(output)
        prediction = prediction.squeeze()
        val, idx = torch.max(prediction, 0)
        return prediction, idx.item()