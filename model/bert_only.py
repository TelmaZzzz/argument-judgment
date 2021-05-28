import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self._weight = nn.Parameter(torch.ones(hidden_dim))
        self._bias = nn.Parameter(torch.zeros(hidden_dim))
        self._eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self._eps)
        return self._weight * x + self._bias


class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self._bert = BertModel.from_pretrained(args.model_name)
        self._FC = nn.Sequential(
            LayerNorm(args.bert_dim),
            nn.Linear(args.bert_dim, args.fc1_dim),
            nn.ELU(inplace=True),
            LayerNorm(args.fc1_dim),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc1_dim, args.fc2_dim),
            nn.ELU(inplace=True),
            LayerNorm(args.fc2_dim),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc2_dim, args.class_num)
        )
        self._F = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.bert_dim, args.class_num),
            # nn.ReLU(inplace=True),
        )

    def forward(self, sen):
        # output = self._FC(self._bert(sen)[1])
        output = self._F(self._bert(sen, return_dict=False)[1])
        return output

