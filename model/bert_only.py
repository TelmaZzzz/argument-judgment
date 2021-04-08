import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self._bert = BertModel.from_pretrained(args.model_name)
        self._FC = nn.Sequential(
            nn.BatchNorm1d(args.bert_dim),
            nn.Linear(args.bert_dim, args.fc1_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.fc1_dim),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc1_dim, args.fc2_dim),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.fc2_dim),
            nn.Dropout(args.dropout),
            nn.Linear(args.fc2_dim, args.class_num)
        )
        self._F = nn.Sequential(
            nn.Linear(args.bert_dim, args.class_num),
            nn.Dropout(args.dropout),
            # nn.ReLU(inplace=True)
        )

    def forward(self, sen):
        # output = self._FC(self._bert(sen)[1])
        output = self._F(self._bert(sen)[1])
        return output

