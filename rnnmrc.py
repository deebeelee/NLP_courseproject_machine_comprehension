import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim
from torch.cuda import FloatTensor

class model(nn.Module):
    """An RNN model that uses two LSTM layers to encode questions and contexts"""
    def __init__(self, vocab_size, hidden_dim=128, out_dim = 2):
        super(model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.ctxt_RNN = nn.LSTM(hidden_dim, 128)
        self.qna_RNN = nn.LSTM(hidden_dim, 128)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(256, out_dim)

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)
        
    def forward(self, context_seq, qna_seq):
        use_cuda = True
        in_context = torch.tensor(context_seq)
        in_question = torch.tensor(qna_seq)
        if use_cuda and torch.cuda.is_available():
            in_context = in_context.cuda()
            in_question = in_question.cuda()
        ctxt_vectors = self.embeds(in_context)
        ctxt_vectors = ctxt_vectors.unsqueeze(1)
        qna_vectors = self.embeds(in_question)
        qna_vectors = qna_vectors.unsqueeze(1)
        _,(output1, _) = self.ctxt_RNN(ctxt_vectors)
        _,(output2, _) = self.qna_RNN(qna_vectors)
        concat = torch.cat([output1.squeeze(), output2.squeeze()], 0)
        output = torch.nn.functional.relu(concat)
        prediction = self.out(output)
        prediction = prediction.squeeze()
        val, idx = torch.max(prediction, 0)
        return prediction, idx.item()