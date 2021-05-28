import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import logging


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


class BERT_ESIM(nn.Module):
    def __init__(self, args):
        super(BERT_ESIM, self).__init__()
        self._bert = BertModel.from_pretrained(args.model_name)
        self._biLSTM = nn.LSTM(args.bert_dim * 4, args.bert_dim // 2, bidirectional=True, dropout=args.dropout, batch_first=True)
        self._FC = nn.Sequential(
            nn.BatchNorm1d(args.bert_dim * 4),
            nn.Linear(args.bert_dim * 4, args.fc1_dim),
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
            nn.Dropout(args.dropout),
            nn.Linear(args.bert_dim * 4, args.bert_dim * 4),
            # nn.ReLU(inplace=True)
        )

    def _pooling(self, v):
        return torch.cat([F.avg_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1), F.max_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1)], -1)

    def _softmax_attention(self, xa, xb):
        e = torch.matmul(xa, xb.transpose(1, 2))
        weightb = e.softmax(-1)
        weighta = e.transpose(1, 2).softmax(-1)
        return torch.matmul(weighta, xa), torch.matmul(weightb, xb)

    def forward(self, sen1, sen2):
        xa, self._xa = self._bert(sen1) # batch * len+2 * dim
        logging.debug("xa shape:{}".format(xa.size()))
        xa = xa.transpose(0, 1)[1:-1]
        xa = xa.transpose(0, 1)
        xb, self._xb = self._bert(sen2)
        logging.debug("xb shape:{}".format(xb.size()))
        xb = xb.transpose(0, 1)[1:-1]
        xb = xb.transpose(0, 1)
        logging.debug("new xa shape:{}".format(xa.size()))
        logging.debug("new xb shape:{}".format(xb.size()))
        ya, yb = self._softmax_attention(xa, xb) # batch * len * dim
        logging.debug("ya shape:{}".format(ya.size()))
        logging.debug("yb shape:{}".format(yb.size()))
        va, _ = self._biLSTM(self._F(torch.cat([xa, ya, xa - ya, xa * ya], -1)))
        vb, _ = self._biLSTM(self._F(torch.cat([xb, yb, xb - yb, xb * yb], -1)))
        logging.debug("va shape:{}".format(va.size()))
        logging.debug("vb shape:{}".format(vb.size()))
        v = torch.cat([self._pooling(va), self._pooling(vb)], -1)
        logging.debug("v shape:{}".format(v.size()))
        terminal = self._FC(v)
        return terminal


class BERT_ESIM_ALL(nn.Module):
    def __init__(self, args):
        super(BERT_ESIM_ALL, self).__init__()
        self._bert = BertModel.from_pretrained(args.model_name)
        self._biLSTM1 = nn.LSTM(args.bert_dim, args.hidden_dim // 2, bidirectional=True, dropout=args.dropout, batch_first=True)
        self._biLSTM2 = nn.LSTM(args.hidden_dim * 4, args.hidden_dim // 2, bidirectional=True, dropout=args.dropout, batch_first=True)
        # self._FC = nn.Linear(args.hidden_dim * 4, args.class_num)
        self._FC = nn.Sequential(
            nn.BatchNorm1d(args.hidden_dim * 4),
            nn.Linear(args.hidden_dim * 4, args.fc1_dim),
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
            # nn.BatchNorm1d(args.hidden_dim * 4),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim * 4, args.hidden_dim * 4),
            # nn.ReLU(inplace=True),
        )
    
    def _f(self, x):
        for index, fuc in enumerate(self._F):
            if index in []:
                x = x.transpose(1, 2)
                x = fuc(x)
                x = x.transpose(1, 2)
            else:
                x = fuc(x)
        return x

    def _softmax_attention(self, xa, xb):
        # xa: batch_size * lena * dim
        # xb: batch_size * lenb * dim
        e = torch.matmul(xa, xb.transpose(1, 2)) # batch_size * lena * lenb
        self.e = e
        weighta = e.softmax(-1) # batch_size * lena * lenb
        weightb = e.transpose(1, 2).softmax(-1) # batch_size * lenb * lena
        logging.debug("weighta:{}".format(weighta.size()))
        logging.debug("weightb:{}".format(weightb.size()))
        return torch.matmul(weighta, xb), torch.matmul(weightb, xa)

    def _pooling(self, v):
        return torch.cat([F.avg_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1), F.max_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1)], -1)
    
    def forward(self, xa, xb):
        xa = self._bert(xa, return_dict=False)[0][:, 1:-1, :]  # batch * len+2 * dim
        logging.debug("xa shape:{}".format(xa.size()))
        # xa = xa.transpose(0, 1)[1:-1]
        # xa = xa.transpose(0, 1)
        # xa = xa[:, 1:-1, :]
        xb = self._bert(xb, return_dict=False)[0][:, 1:-1, :]
        logging.debug("xb shape:{}".format(xb.size()))
        # xb = xb.transpose(0, 1)[1:-1]
        # xb = xb.transpose(0, 1)
        # xb = xb[:, 1:-1, :]
        xa, _ = self._biLSTM1(xa)
        xb, _ = self._biLSTM1(xb)
        logging.debug("xa in LSTM1:{}".format(xa.size()))
        logging.debug("xb in LSTM2:{}".format(xb.size()))

        # ya: batch_size * lena * dim
        # yb: batch_size * lenb * dim
        ya, yb = self._softmax_attention(xa, xb)
        logging.debug("ya:{}".format(ya.size()))
        va, _ = self._biLSTM2(self._f(torch.cat([xa, ya, xa - ya, xa * ya], -1)))
        vb, _ = self._biLSTM2(self._f(torch.cat([xb, yb, xb - yb, xb * yb], -1)))
        # va: batch_size * lena * dim
        # vb: batch_size * lenb * dim
        logging.debug("size va:{}".format(va.size()))

        v = torch.cat([self._pooling(va), self._pooling(vb)], -1)
        logging.debug("size v:{}".format(v.size()))
        terminal = self._FC(v)
        logging.debug("terminal:{}".format(terminal.size()))
        return terminal
