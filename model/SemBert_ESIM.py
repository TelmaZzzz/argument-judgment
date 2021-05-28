import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import logging


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self._char_dim = args.bert_dim
        self._filter_size = args.filter_size
        self._char_cnn = nn.Conv1d(self._char_dim, self._char_dim, kernel_size=self._filter_size,
                                   padding=0)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        batch_size, max_seq_len, max_word_len, dim = inputs.size()
        inputs = inputs.view(-1, max_word_len, dim)
        x = inputs.transpose(1, 2)
        logging.debug("bf cnn size:{}".format(x.shape))
        x = self._char_cnn(x)
        x = self._relu(x)
        logging.debug("af cnn size:{}".format(x.shape))
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        logging.debug("af pool size:{}".format(x.shape))
        x = self._dropout(x.squeeze())
        return x.view(batch_size, max_seq_len, -1)


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


class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self._embed = nn.Embedding(args.tag_vocab_size, args.tag_hidden_size, padding_idx=0)
        self._LN = LayerNorm(args.tag_hidden_size)
        self._dropout = nn.Dropout(args.dropout)

    def forward(self, inputs_tag_ids):
        x = self._embed(inputs_tag_ids)
        x = self._LN(x)
        x = self._dropout(x)
        return x


class TagEmbeddings(nn.Module):
    def __init__(self, args):
        super(TagEmbeddings, self).__init__()
        self._hidden_size = args.tag_hidden_size
        self._num_aspect = args.max_aspect
        self._embed = Embeddings(args)
        self._fc = nn.Linear(self._hidden_size, args.tag_output_dim)
        self._dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        embed = self._embed(inputs)
        embed = self._dropout(embed)
        input = embed.view(-1, self._num_aspect, inputs.size(1), self._hidden_size)
        logit = self._fc(input)
        return logit


class Tag_layer(nn.Module):
    def __init__(self, args):
        super(Tag_layer, self).__init__()
        self._embed = Embeddings(args)
        self._hidden_dim = args.tag_hidden_size
        self._bigru = nn.GRU(args.tag_hidden_size, args.tag_hidden_size,
                             num_layers=args.tag_num_layer,
                             bidirectional=True, batch_first=True)
        self._fc = nn.Linear(args.tag_hidden_size * 2 * args.max_aspect, args.tag_output_dim)
        self._dropout = nn.Dropout(args.dropout)
        self._max_aspect = args.max_aspect

    def forward(self, inputs_tag_ids):
        batch_size_aspect, max_seq_len = inputs_tag_ids.size()
        embed = self._embed(inputs_tag_ids)
        x = embed.view(batch_size_aspect, embed.size(1), -1)
        x, _ = self._bigru(x)
        x = x.view(-1, self._max_aspect, max_seq_len, 2 * self._hidden_dim)
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, max_seq_len, -1)
        x = self._fc(x)
        return x


class SemBert(nn.Module):
    def __init__(self, args):
        super(SemBert, self).__init__()
        self._cnn = CNN(args)
        self._bert = BertModel.from_pretrained(args.model_name)
        self._tag_flag = True
        if self._tag_flag:
            self._tag_layer = Tag_layer(args)
        else:
            self._tag_layer = TagEmbeddings(args)
            self._dence = nn.Linear(args.max_aspect * args.tag_output_dim, args.tag_output_dim)
        self._filter_size = args.filter_size

    def forward(self, inputs, inputs_mask, inputs_tag, inputs_word_start_end):
        logging.debug("inputs size:{}".format(inputs.size()))
        inputs, _ = self._bert(inputs, attention_mask=inputs_mask, return_dict=False)
        # return inputs[:, 1:, :]
        logging.debug("inputs after bert:{}".format(inputs))
        batch_size, seq_len, dim = inputs.size()
        # logging.debug("batch_size:{} seq_len:{} dim:{}".format(batch_size, seq_len, dim))
        # max_word_len = self._filter_size
        # max_seq_len = 0
        # for batch_item in inputs_word_start_end:
        #     word = 0
        #     for start, end in batch_item:
        #         if start != -1 and end != -1:
        #             max_word_len = max(max_word_len, end - start + 1)
        #             word += 1
        #     max_seq_len = max(max_seq_len, word)
        # batch_start_end = []
        # for ids, batch in enumerate(inputs_word_start_end):
        #     offset = ids * seq_len
        #     word_seq = []
        #     for start, end in batch:
        #         if start != -1 and end != -1:
        #             subword_list = list(range(offset + start + 1, offset + end + 2))
        #             subword_list += [0] * (max_word_len - len(subword_list))
        #             word_seq.append(subword_list)
        #     while len(word_seq) < max_seq_len:
        #         word_seq.append([0] * max_word_len)
        #     batch_start_end.append(word_seq)
        # batch_start_end = torch.tensor(batch_start_end)
        # batch_start_end = batch_start_end.view(-1)
        # inputs = inputs.view(-1, dim)
        # inputs = torch.cat([inputs.new_zeros((1, dim)), inputs], dim=0)
        # batch_start_end = batch_start_end.cuda()
        # cnn_bert = inputs.index_select(0, batch_start_end)
        # logging.debug("cnn_bert size:{}".format(cnn_bert.shape))
        # cnn_bert = cnn_bert.view(batch_size, max_seq_len, max_word_len, dim)
        # bert_output = self._cnn(cnn_bert)
        # bert_output = bert_output.view(batch_size, max_seq_len, -1)
        # logging.debug("tag_ids size:{}".format(inputs_tag.size()))
        inputs_tag = inputs_tag[:, :, :seq_len]
        inputs_tag_ids = inputs_tag.view(-1, seq_len)
        if self._tag_flag is True:
            tag_output = self._tag_layer(inputs_tag_ids)
            # logging.info("YES")
        else:
            tag_output = self._tag_layer(inputs_tag_ids)
            # logging.info("tag_output:{}".format(tag_output.shape))
            tag_output = tag_output.transpose(1, 2).contiguous().view(batch_size,
                                                                      seq_len, -1)
            tag_output = self._dence(tag_output)
        # logging.debug("tag_output:{}".format(tag_output.shape))
        # logging.info("inputs:{}".format(inputs.shape))
        # logging.info("tag_output:{}".format(tag_output.shape))
        bert_output = torch.cat([inputs, tag_output], dim=2)
        logging.debug("bert_output size:{}".format(bert_output.shape))
        # bert_output.view(batch_size, max_seq_len, dim)

        return bert_output[:, 1:, :]


class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self._embed = SemBert(args)
        self._fc = nn.Linear(args.bert_dim + args.tag_output_dim, args.hidden_dim)
        self._biLSTM1 = nn.LSTM(args.bert_dim + args.tag_output_dim, args.hidden_dim // 2, bidirectional=True, dropout=args.dropout, batch_first=True)
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
        e = torch.matmul(xa, xb.transpose(1, 2))  # batch_size * lena * lenb
        self.e = e
        weighta = e.softmax(-1)  # batch_size * lena * lenb
        weightb = e.transpose(1, 2).softmax(-1)  # batch_size * lenb * lena
        logging.debug("weighta:{}".format(weighta.size()))
        logging.debug("weightb:{}".format(weightb.size()))
        return torch.matmul(weighta, xb), torch.matmul(weightb, xa)

    def _pooling(self, v):
        return torch.cat([F.avg_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1), F.max_pool1d(v.transpose(1, 2), v.size(1)).squeeze(-1)], -1)

    def forward(self, inputs_a, inputs_a_mask, inputs_a_tag_ids, inputs_a_word_start_end,
                inputs_b, inputs_b_mask, inputs_b_tag_ids, inputs_b_word_start_end):
        xa = self._embed(inputs_a, inputs_a_mask, inputs_a_tag_ids, inputs_a_word_start_end)
        xb = self._embed(inputs_b, inputs_b_mask, inputs_b_tag_ids, inputs_b_word_start_end)
        xa = self._fc(xa)
        xb = self._fc(xb)
        # xa, _ = self._biLSTM1(xa)
        # xb, _ = self._biLSTM1(xb)
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
