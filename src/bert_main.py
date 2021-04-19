# -*- coding: utf-8
import torch
import os
import sys
import json
import jsonlines
import importlib
importlib.reload(sys)
import codecs
# sys.setdefaultencoding( "utf-8" )
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach()) 
sys.path.append(os.getcwd())
sys.path.append("/users10/lyzhang/opt/tiger/argument-judgment")
from util import common
from model import bert_only, bert_ESIM
import train
import random

import warnings
import logging
import argparse
from transformers import BertModel, BertTokenizer
from ltp import LTP
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")

class Example(object):
    def __init__(self, sen1, sen2, label):
        self.sen1 = sen1
        self.sen2 = sen2
        self.label = label

class Feature(object):
    def __init__(self, inputs, inputs_word_start_end, inputs_mask, inputs_sen_ids, sen1_srl, sen2_srl, inputs_srl_ids, label_ids):
        self.inputs = inputs
        self.inputs_word_start_end = inputs_word_start_end
        self.inputs_mask = inputs_mask
        self.inputs_sen_ids = inputs_sen_ids
        self.inputs_srl_ids = inputs_srl_ids
        self.sen1_srl = sen1_srl
        self.sen2_srl = sen2_srl
        self.label_ids = label_ids

def main(args):
    logging.info("Start prepare data...")
    train_data = common.load_data(args.train_path)
    valid_data = common.load_data(args.valid_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    _, label_vocab = common.build_vocab(train_data)
    train_sens = torch.LongTensor([common.bert_concat_tokenizer(iter["sen1"], iter["sen2"], tokenizer, args.fix_length) for iter in train_data])
    train_labels = torch.LongTensor([label_vocab[1][iter["label"]] for iter in train_data])
    valid_sens = torch.LongTensor([common.bert_concat_tokenizer(iter["sen1"], iter["sen2"], tokenizer, args.fix_length) for iter in valid_data])
    valid_labels = torch.LongTensor([label_vocab[1][iter["label"]] for iter in valid_data])
    train_dataset = torch.utils.data.TensorDataset(train_sens, train_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid_sens, valid_labels)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
    logging.info("Success preparing data!!!")
    args.class_num = len(label_vocab[0])
    model = bert_only.BERT(args).cuda()
    train.bert_train(train_iter, valid_iter, model, args)

def bert_ESIM_main(args):
    logging.info("Start prepare data...")
    train_data = common.load_data(args.train_path)
    valid_data = common.load_data(args.valid_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    _, label_vocab = common.build_vocab(train_data)
    train_sen1 = torch.LongTensor([common.bert_tokenizer(iter["sen1"], tokenizer, args.fix_length) for iter in train_data])
    train_sen2 = torch.LongTensor([common.bert_tokenizer(iter["sen2"], tokenizer, args.fix_length) for iter in train_data])
    train_label = torch.LongTensor([label_vocab[1][iter["label"]] for iter in train_data])
    valid_sen1 = torch.LongTensor([common.bert_tokenizer(iter["sen1"], tokenizer, args.fix_length) for iter in valid_data])
    valid_sen2 = torch.LongTensor([common.bert_tokenizer(iter["sen2"], tokenizer, args.fix_length) for iter in valid_data])
    valid_label = torch.LongTensor([label_vocab[1][iter["label"]] for iter in valid_data])
    train_dataset = torch.utils.data.TensorDataset(train_sen1, train_sen2, train_label)
    valid_dataset = torch.utils.data.TensorDataset(valid_sen1, valid_sen2, valid_label)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
    logging.info("Finished Preparing data!!!")
    args.class_num = len(label_vocab[0])
    # model = bert_ESIM.BERT_ESIM(args).cuda()
    model = bert_ESIM.BERT_ESIM_ALL(args).cuda()
    train.two_sen_train(train_iter, valid_iter, model, args)

def load_data(path):
    Examples = []
    with open(path, "r", encoding="utf-8") as f:
        raw_data = jsonlines.Reader(f)
        for item in raw_data:
            Examples.append(Example(item["sen1", item["sen2"], item["label"]]))
    return Examples

def get_tag(sen, ltp):
    seg, hidden = ltp.seg([sen.split("|")], is_preseged=True)
    seg_srl = ltp.srl(hidden, keep_empty=False)
    srl_results = []
    for items in seg_srl:
        for item in items:
            result = ["O"] * (len(seg[0]) + 1)
            result[item[0]+1] = "Verb"
            for tag in item[1]:
                for i in range(tag[1]+1, tag[2]+2):
                    result[i] = tag[0]
                srl_results.append(result)
    return seg[0], srl_results

def _truncate_sen_pair(token1, token1_word_ids, token2, token2_word_ids, max_length):
    while True:
        if len(token1) + len(token2) <= max_length:
            break
        elif len(token1) > len(token2):
            token1.pop()
            token1_word_ids.pop()
        elif len(token1) <= len(token2):
            token2.pop()
            token2_word_ids.pop()

def gen_feature(raw_data, tokenizer, ltp, label_vocab, args):
    Features = []
    for sen_ids, data in enumerate(raw_data):
        token1 = []
        token2 = []
        token1_raw, sen1_srl = get_tag(data.sen1, ltp)
        if args.max_aspect < len(sen1_srl):
            args.max_aspect = len(sen1_srl)
        token1_ids = []
        for ids, word in enumerate(token1_raw):
            word_token = tokenizer.tokenize(word)
            token1 += word_token
            token1_ids += [ids + 1] * len(word_token)
        token2_raw, sen2_srl = get_tag(data.sen2, ltp)
        if args.max_aspect < len(sen2_srl):
            args.max_aspect = len(sen2_srl)
        token2_ids = []
        for ids, word in enumerate(token2_raw):
            word_token = tokenizer.tokenize(word)
            token2 += word_token
            token2_ids += [ids +1] * len(word_token)
        if len(token1) + len(token2) > args.max_length - 3:
            logging.info("sentence {} is Too Long!!!".format(sen_ids))
        _truncate_sen_pair(token1, token1_ids, token2, token2_ids, args.max_length)
        inputs_tokens = ["[CLS]"] + token1 + ["[SEP]"] + token2 + ["[SEP]"]
        sen1_srl = sen1_srl[:][:token1_ids[-1]]
        sen2_srl = sen2_srl[:][:token2_ids[-1]]
        inputs = tokenizer.convert_tokens_to_ids(inputs_tokens)
        inputs_word_ids = [0] + token1_ids + [0] + token2_ids + [0]
        inputs_sen_ids = [0] * (len(token1) + 2)
        inputs_sen_ids += [1] * (len(token2) + 1)
        inputs_mask = [1] * len(inputs)
        padding = [0] * (args.max_length - len(inputs))
        inputs += padding
        inputs_sen_ids += padding
        inputs_mask += padding
        start = -1
        pre_word = -1
        word_start_end = []
        # logging.info("inputs_word_ids:{}".format(inputs_word_ids))
        for ids, word_ids in enumerate(inputs_word_ids):
            end = ids
            # logging.info("{} : {}".format(pre_word, ids))
            if pre_word != word_ids:
                if start != -1:
                    word_start_end.append((start, end-1))
                start = ids
            pre_word = word_ids
        if start != -1:
            word_start_end.append((start, end))
        word_start_end += [(-1, -1)] * (args.max_length - len(word_start_end))
        label_ids = label_vocab[0][data.label]
        Features.append(
            Feature(
                inputs=inputs,
                inputs_word_start_end=word_start_end,
                inputs_mask=inputs_mask,
                inputs_sen_ids=inputs_sen_ids,
                sen1_srl=sen1_srl,
                sen2_srl=sen2_srl,
                inputs_srl_ids=None,
                label_ids=label_ids
            )
        )
    return Features

def build_srl_vocab(features):
    srl_vocab = dict()
    stoi = dict()
    itos = ["<PAD>", "<CLS>", "<SEP>"]
    for feature in features:
        for words in feature.sen1_srl:
            for word in words:
            # logging.info(word)
                if srl_vocab.get(word, None) is not None:
                    srl_vocab[word] += 1
                else:
                    srl_vocab[word] = 1
        for words in feature.sen2_srl:
            for word in words:
                if srl_vocab.get(word, None) is not None:
                    srl_vocab[word] += 1
                else:
                    srl_vocab[word] = 1
    sort_result = sorted(srl_vocab.items(), key=lambda item: item[1], reverse=True)
    for key, _ in sort_result:
        itos.append(key)
    for ids, key in enumerate(itos):
        stoi[key] = ids
    return stoi, itos

def get_copy_srl(srl_list):
    max_cnt = 0
    max_ids = -1
    for ids, srl in enumerate(srl_list):
        cnt = 0
        for word in srl:
            if word != 'O':
                cnt += 1
            if max_cnt < cnt:
                max_cnt = cnt
                max_ids = ids
    return srl_list[max_ids]

def convert_tag_ids(sen, vocab):
    sen_ids = []
    for word in sen:
        sen_ids.append(vocab.get(word, vocab['O']))
    return sen_ids

def transform_srl_feature(features, vocab, args):
    Features = []
    for feature in features:
        if len(feature.sen1_srl) < args.max_aspect:
            copy_srl = get_copy_srl(feature.sen1_srl)
            while len(feature.sen1_srl) < args.max_aspect:
                feature.sen1_srl.append(copy_srl.copy())
        if len(feature.sen2_srl) < args.max_aspect:
            copy_srl = get_copy_srl(feature.sen2_srl)
            while len(feature.sen2_srl) < args.max_aspect:
                feature.sen2_srl.append(copy_srl.copy())
        
        assert len(feature.sen1_srl) == args.max_aspect
        assert len(feature.sen2_srl) == args.max_aspect
        
        inputs_tag_ids = []
        for sen1, sen2 in zip(feature.sen1_srl, feature.sen2_srl):
            inputs_tag_ids.append([1] + convert_tag_ids(sen1, vocab) + [2] + convert_tag_ids(sen2, vocab) + [2])
            inputs_tag_ids[-1] = inputs_tag_ids[-1] + [0] * (args.max_length - len(inputs_tag_ids[-1]))
        feature.inputs_tag_ids = inputs_tag_ids
        Features.append(feature)
    return Features

def Sembert_ESIM_main(args):
    logging.info("Start prepare data...")
    train_data = load_data(args.train_path)
    valid_data = load_data(args.valid_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    ltp = LTP()
    label_vocab = (["A2B", "B2A", "Neutral"], {"A2B":0, "B2A":1, "Neutral":2})
    train_features = gen_feature(train_data, tokenizer, ltp, label_vocab, args)
    stoi, itos = build_srl_vocab(train_features)
    train_features = transform_srl_feature(train_features, stoi, args)

def test(args):
    sen1 = "这|就|是|语言|促进|沟通|的|力量|呀|！"
    sen2 = "它|是|沟通|人|与|人|之间|的|桥梁|，|它|是|化解|仇恨|的|有用|手段|，|它|是|国|与|国|合作|的|基础|……"
    label = "A2B"
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    ltp = LTP()
    Examples = [Example(sen1, sen2, label)]
    label_vocab = ({"A2B":0, "B2A":1, "Neutral":2}, ["A2B", "B2A", "Neutral"])
    features = gen_feature(Examples, tokenizer, ltp, label_vocab, args)
    srl_stoi, srl_itos = build_srl_vocab(features)
    logging.info(srl_itos)
    features = transform_srl_feature(features, srl_stoi, args)
    for feature in features:
        logging.info("inputs:{}".format(feature.inputs))
        logging.info("inputs mask:{}".format(feature.inputs_mask))
        logging.info("inputs sen ids:{}".format(feature.inputs_sen_ids))
        logging.info("inputs tag ids:{}".format(feature.inputs_tag_ids))
        logging.info("label ids:{}".format(feature.label_ids))
        # for st, en in feature.inputs_word_start_end:
        #     print(st, en)
        # logging.info("inputs word start end:{}".foramt(feature.word_start_end))
    
    # model.train()
    # logit = model(sen1, sen2)
    # loss = torch.nn.functional.cross_entropy(logit.cpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="data/train.json")
    parser.add_argument("--valid-path", type=str, default="data/valid.json")
    parser.add_argument("--model-name", type=str, default="bert-base-chinese")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--fix-length", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.000001)
    parser.add_argument("--hidden-dim", type=int, default=600)
    parser.add_argument("--bert-dim", type=int, default=768)
    parser.add_argument("--fc1-dim", type=int, default=354)
    parser.add_argument("--fc2-dim", type=int, default=132)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train-type", type=str, default="bert-only")
    parser.add_argument("--max-aspect", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=60)
    args = parser.parse_args()
    if args.train_type == "bert-only":
        main(args)
    elif args.train_type == "bert-ESIM":
        bert_ESIM_main(args)
    elif args.train_type == "test":
        test(args)