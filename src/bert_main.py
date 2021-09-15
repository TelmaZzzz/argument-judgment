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
from model.SemBert_ESIM import SemBert, ESIM
import train
import random

import threading
import warnings
import logging
import argparse
import numpy as np
from transformers import BertModel, BertTokenizer
from ltp import LTP
from torch.utils.data import TensorDataset, DataLoader
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


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
    logging.info(label_vocab[0])
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
    if args.mode == "predict":
        model = torch.load(args.model_load_path).cuda()
        test_data = common.load_data(args.test_path)
        test_sens = torch.LongTensor([common.bert_concat_tokenizer(iter["sen1"], iter["sen2"], tokenizer, args.fix_length) for iter in test_data])
        test_labels = torch.LongTensor([label_vocab[1][iter["label"]] for iter in test_data])
        test_dataset = torch.utils.data.TensorDataset(test_sens, test_labels)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
        train.bert_eval(test_iter, model, args)
    else:    
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
    if args.mode == "predict":
        model = torch.load(args.model_load_path).cuda()
        test_data = common.load_data(args.test_path)
        test_sen1 = torch.LongTensor([common.bert_tokenizer(iter["sen1"], tokenizer, args.fix_length) for iter in test_data])
        test_sen2 = torch.LongTensor([common.bert_tokenizer(iter["sen2"], tokenizer, args.fix_length) for iter in test_data])
        test_label = torch.LongTensor([label_vocab[1][iter["label"]] for iter in test_data])
        test_dataset = torch.utils.data.TensorDataset(test_sen1, test_sen2, test_label)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
        _ = train.two_sen_eval(test_iter, model, args)
        logging.info("END")
    else:
        model = bert_ESIM.BERT_ESIM_ALL(args).cuda()
        train.two_sen_train(train_iter, valid_iter, model, args)


def load_data(path):
    Examples = []
    with open(path, "r", encoding="utf-8") as f:
        raw_data = jsonlines.Reader(f)
        for item in raw_data:
            Examples.append(Example(item["sen1"], item["sen2"], item["label"]))
    return Examples


def get_tag(sen, ltp):
    seg, hidden = ltp.seg([sen])
    try:
        seg_srl = ltp.srl(hidden, keep_empty=False)
    except Exception as e:
        logging.info(seg)
        logging.info(sen)
        logging.info(len(seg[0]))
        print(sen)
        logging.error(e)

    srl_results = []
    for items in seg_srl:
        for item in items:
            result = ["O"] * (len(seg[0]))
            result[item[0]] = "Verb"
            for tag in item[1]:
                for i in range(tag[1], tag[2]+1):
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


def gen_feature_v2(raw_data, label_vocab, args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    ltp = LTP()
    Features = []
    for sen_ids, data in enumerate(raw_data):
        if sen_ids % 500 == 0:
            logging.info("sen_ids:{}".format(sen_ids))
        token1 = []
        token2 = []
        try:
            token1_raw, sen1_srl = get_tag(data.sen1, ltp)
        except Exception as e:
            logging.warning("sen_id:{} have some mistake.".format(sen_ids))
            logging.error(e)
            continue
        if len(sen1_srl) == 0:
            continue
        if args.max_aspect < len(sen1_srl):
            args.max_aspect = len(sen1_srl)
        token1_ids = [0]
        for ids, word in enumerate(token1_raw):
            word_token = tokenizer.tokenize(word)
            token1 += word_token
            token1_ids += [ids + 1] * len(word_token)
        try:
            token2_raw, sen2_srl = get_tag(data.sen2, ltp)
        except Exception as e:
            logging.warning("sen_id:{} have some mistake.".format(sen_ids))
            logging.error(e)
            continue
        if len(sen2_srl) == 0:
            continue
        if args.max_aspect < len(sen2_srl):
            args.max_aspect = len(sen2_srl)
        token2_ids = [0]
        for ids, word in enumerate(token2_raw):
            word_token = tokenizer.tokenize(word)
            token2 += word_token
            token2_ids += [ids + 1] * len(word_token)
        while len(token1) > args.max_length - 2:
            token1.pop()
            token1_ids.pop()
        while len(token2) > args.max_length - 2:
            token2.pop()
            token2_ids.pop()
        # logging.info("sen1 size:{}. token1_ids:{} type:{}".format(len(sen1_srl[0]), token1_ids[-1], type(token1_ids[-1])))
        for i, sen in enumerate(sen1_srl):
            sen1_srl[i] = sen[:token1_ids[-1]]
        for i, sen in enumerate(sen2_srl):
            sen2_srl[i] = sen[:token2_ids[-1]]
        # sen1_srl = sen1_srl[:, :token1_ids[-1]]
        # sen2_srl = sen2_srl[:, :token2_ids[-1]]
        # logging.info("token1_ids:{}".format(token1_ids[-1]))
        # logging.info("token2_ids:{}".format(token2_ids[-1]))
        assert len(sen1_srl[0]) <= args.max_length
        assert len(sen2_srl[0]) <= args.max_length
        inputs_token1 = ["[CLS]"] + token1 + ["[SEP]"]
        inputs_token1 = tokenizer.convert_tokens_to_ids(inputs_token1)
        inputs_token2 = ["[CLS]"] + token2 + ["[SEP]"]
        inputs_token2 = tokenizer.convert_tokens_to_ids(inputs_token2)
        token1_ids.append(0)
        token2_ids.append(0)
        inputs_mask_1 = [1] * len(inputs_token1)
        inputs_mask_2 = [1] * len(inputs_token2)
        padding_1 = [0] * (args.max_length - len(inputs_token1))
        padding_2 = [0] * (args.max_length - len(inputs_token2))
        inputs_token1 += padding_1
        inputs_token2 += padding_2
        inputs_mask_1 += padding_1
        inputs_mask_2 += padding_2
        start = -1
        pre_word = -1
        word_start_end_1 = []
        for ids, word_ids in enumerate(token1_ids):
            end = ids
            # logging.info("{} : {}".format(pre_word, ids))
            if pre_word != word_ids:
                if start != -1:
                    word_start_end_1.append((start, end-1))
                start = ids
            pre_word = word_ids
        if start != -1:
            word_start_end_1.append((start, end))
        word_start_end_1 += [(-1, -1)] * (args.max_length - len(word_start_end_1))
        start = -1
        pre_word = -1
        word_start_end_2 = []
        for ids, word_ids in enumerate(token2_ids):
            end = ids
            # logging.info("{} : {}".format(pre_word, ids))
            if pre_word != word_ids:
                if start != -1:
                    word_start_end_2.append((start, end-1))
                start = ids
            pre_word = word_ids
        if start != -1:
            word_start_end_2.append((start, end))
        word_start_end_2 += [(-1, -1)] * (args.max_length - len(word_start_end_2))
        label_ids = label_vocab[0][data.label]
        Features.append(
            Feature(
                inputs=(inputs_token1, inputs_token2),
                inputs_word_start_end=(word_start_end_1, word_start_end_2),
                inputs_mask=(inputs_mask_1, inputs_mask_2),
                inputs_sen_ids=None,
                sen1_srl=sen1_srl,
                sen2_srl=sen2_srl,
                inputs_srl_ids=None,
                label_ids=label_ids
            )
        )
    return Features


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
            token2_ids += [ids + 1] * len(word_token)
        if len(token1) + len(token2) > args.max_length - 3:
            logging.info("sentence {} is Too Long!!!".format(sen_ids))
        _truncate_sen_pair(token1, token1_ids, token2, token2_ids, args.max_length)
        inputs_tokens = ["[CLS]"] + token1 + ["[SEP]"] + token2 + ["[SEP]"]
        sen1_srl = sen1_srl[:, :token1_ids[-1]]
        sen2_srl = sen2_srl[:, :token2_ids[-1]]
        # assert len(sen1_srl[0]) <= args.max_length
        # assert len(sen2_srl[0]) <= args.max_length
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
    max_ids = 0
    for ids, srl in enumerate(srl_list):
        cnt = 0
        for word in srl:
            if word != 'O':
                cnt += 1
        if max_cnt < cnt:
            max_cnt = cnt
            max_ids = ids
    try:
        srl_list[max_ids]
    except Exception:
        logging.info("max_ids:{}".format(max_ids))
        logging.info(srl_list)
    return srl_list[max_ids]


def convert_tag_ids(sen, vocab):
    sen_ids = []
    for word in sen:
        sen_ids.append(vocab.get(word, vocab['O']))
    return sen_ids


def convert_tag_ids_v2(sen, vocab, word_start_end):
    sen_ids = []
    for idx, word in enumerate(sen):
        sen_ids += [vocab.get(word, vocab['O'])] * (word_start_end[idx][1] - word_start_end[idx][0] + 1)
    return sen_ids


def transform_srl_feature_v3(features, vocab, args):
    Features = []
    for idx, feature in enumerate(features):
        if idx % 500 == 0:
            logging.info("transform sen_ids:{}".format(idx))
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

        inputs_tag_ids_1 = []
        inputs_tag_ids_2 = []
        for sen in feature.sen1_srl:
            inputs_tag_ids_1.append(convert_tag_ids_v2(['<CLS>'] + sen + ['<SEP>'], vocab, feature.inputs_word_start_end[0]))
            inputs_tag_ids_1[-1] = inputs_tag_ids_1[-1] + [0] * (args.max_length - len(inputs_tag_ids_1[-1]))
        for sen in feature.sen2_srl:
            inputs_tag_ids_2.append(convert_tag_ids_v2(['<CLS>'] + sen + ['<SEP>'], vocab, feature.inputs_word_start_end[1]))
            inputs_tag_ids_2[-1] = inputs_tag_ids_2[-1] + [0] * (args.max_length - len(inputs_tag_ids_2[-1]))
        feature.inputs_tag_ids = (inputs_tag_ids_1, inputs_tag_ids_2)
        Features.append(feature)
    return Features


def transform_srl_feature_v2(features, vocab, args):
    Features = []
    for idx, feature in enumerate(features):
        if idx % 500 == 0:
            logging.info("transform sen_ids:{}".format(idx))
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

        inputs_tag_ids_1 = []
        inputs_tag_ids_2 = []
        for sen in feature.sen1_srl:
            inputs_tag_ids_1.append([1] + convert_tag_ids(sen, vocab) + [2])
            inputs_tag_ids_1[-1] = inputs_tag_ids_1[-1] + [0] * (args.max_length - len(inputs_tag_ids_1[-1]))
        for sen in feature.sen2_srl:
            inputs_tag_ids_2.append([1] + convert_tag_ids(sen, vocab) + [2])
            inputs_tag_ids_2[-1] = inputs_tag_ids_2[-1] + [0] * (args.max_length - len(inputs_tag_ids_2[-1]))
        feature.inputs_tag_ids = (inputs_tag_ids_1, inputs_tag_ids_2)
        Features.append(feature)
    return Features


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


def build_dataset(features):
    test_sen1 = torch.LongTensor([feature.inputs[0] for feature in features])
    test_sen2 = torch.LongTensor([feature.inputs[1] for feature in features])
    test_sen1_mask = torch.LongTensor([feature.inputs_mask[0] for feature in features])
    test_sen2_mask = torch.LongTensor([feature.inputs_mask[1] for feature in features])
    test_sen1_tag = torch.LongTensor([feature.inputs_tag_ids[0] for feature in features])
    test_sen2_tag = torch.LongTensor([feature.inputs_tag_ids[1] for feature in features])
    test_sen1_start_end = torch.LongTensor([feature.inputs_word_start_end[0] for feature in features])
    test_sen2_start_end = torch.LongTensor([feature.inputs_word_start_end[1] for feature in features])
    test_label = torch.LongTensor([feature.label_ids for feature in features])
    dataset = TensorDataset(test_sen1, test_sen1_mask, test_sen1_tag, test_sen1_start_end,
                            test_sen2, test_sen2_mask, test_sen2_tag, test_sen2_start_end,
                            test_label)
    return dataset


def load_dataset(path, args):
    with np.load(path) as data:
        test_sen1 = torch.LongTensor(data["sen1"].tolist())
        test_sen2 = torch.LongTensor(data["sen2"].tolist())
        test_sen1_mask = torch.LongTensor(data["sen1_mask"].tolist())
        test_sen2_mask = torch.LongTensor(data["sen2_mask"].tolist())
        test_sen1_tag = torch.LongTensor(data["sen1_tag"].tolist())
        test_sen2_tag = torch.LongTensor(data["sen2_tag"].tolist())
        test_sen1_start_end = torch.LongTensor(data["sen1_start_end"].tolist())
        test_sen2_start_end = torch.LongTensor(data["sen2_start_end"].tolist())
        test_label = torch.LongTensor(data["label"].tolist())
        args.max_aspect = max(args.max_aspect, test_sen1_tag.size(1))
    logging.info(test_sen1_tag.size())
    logging.info(test_sen2_tag.size())
    dataset = TensorDataset(test_sen1, test_sen1_mask, test_sen1_tag, test_sen1_start_end,
                            test_sen2, test_sen2_mask, test_sen2_tag, test_sen2_start_end,
                            test_label)
    return dataset


def Sembert_ESIM_main(args):
    logging.info("Start prepare data...")
    logging.info("train save path:{}".format(args.train_save_path))
    if args.train_load_path is None:
        train_data = load_data(args.train_path)
        valid_data = load_data(args.valid_path)
        # train_data = train_data[:int(len(train_data) * 0.1)]
        # valid_data = valid_data[:int(len(valid_data) * 0.1)]
        label_vocab = ({"A2B": 0, "B2A": 1, "Neutral": 2}, ["A2B", "B2A", "Neutral"])
        args.class_num = 3
        threads = []
        for i in range(4):
            thread = MyThread(gen_feature_v2, args=(train_data[int(len(train_data) * (0.25 * i)):int(len(train_data) * (0.25 * (i + 1)))],
                              label_vocab, args))
            threads.append(thread)
        thread_valid = MyThread(gen_feature_v2, args=(valid_data,
                                label_vocab, args))
        for thread in threads:
            thread.start()
        thread_valid.start()
        for thread in threads:
            thread.join()
        thread_valid.join()
        train_features = []
        for thread in threads:
            train_features += thread.get_result()
        logging.info("train_features size:{}".format(len(train_features)))
        # train_features = gen_feature_v2(train_data[:int(len(train_data) * 0.01)], tokenizer, ltp, label_vocab, args)
        # valid_features = gen_feature_v2(valid_data, label_vocab, args)
        valid_features = thread_valid.get_result()
        stoi, itos = build_srl_vocab(train_features)
        logging.info(stoi)
        logging.info(itos)
        args.tag_vocab_size = len(itos)
        threads = []
        for i in range(4):
            thread = MyThread(transform_srl_feature_v2, args=(train_features[int(len(train_features) * (0.25 * i)):int(len(train_features) * (0.25 * (i + 1)))],
                              stoi, args))
            threads.append(thread)
        thread_valid = MyThread(transform_srl_feature_v2, args=(valid_features,
                                stoi, args))
        for thread in threads:
            thread.start()
        thread_valid.start()
        for thread in threads:
            thread.join()
        thread_valid.join()
        train_features = []
        for thread in threads:
            train_features += thread.get_result()
        # train_features = transform_srl_feature_v2(train_features, stoi, args)
        # valid_features = transform_srl_feature_v2(valid_features, stoi, args)
        valid_features = thread_valid.get_result()
        logging.info("train_features size:{}".format(len(train_features)))
        logging.info("valid_features size:{}".format(len(valid_features)))
        logging.info("Finished prepare raw data...")
        train_dataset = build_dataset(train_features)
        valid_dataset = build_dataset(valid_features)
    else:
        train_dataset = load_dataset(args.train_load_path, args)
        valid_dataset = load_dataset(args.valid_load_path, args)
        itos = ['<PAD>', '<CLS>', '<SEP>', 'O', 'A1', 'A0', 'Verb', 'ARGM-ADV', 'ARGM-TMP', 'A2', 'ARGM-DIS', 'ARGM-LOC', 'ARGM-PRP', 'ARGM-MNR', 'ARGM-CND', 'ARGM-DIR', 'ARGM-TPC', 'ARGM-BNF', 'ARGM-EXT', 'A0-CRD', 'A0-PSR', 'A3', 'A0-PSE', 'ARGM-FRQ', 'A0-PRD', 'A1-QTY', 'A1-CRD', 'A1-PSR', 'A1-PSE', 'ARGM-DGR', 'A1-PRD', 'ARGM-PRD', 'A4']
    args.class_num = 3
    args.tag_vocab_size = len(itos)
    logging.info("Finished prepare dataset...")
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_iter = DataLoader(valid_dataset, batch_size=args.batch_size)
    logging.info("Finished prepare dataloader...")
    if args.mode == "predict":
        model = torch.load(args.model_load_path).cuda()
        test_dataset = load_dataset(args.test_load_path, args)
        test_iter = DataLoader(test_dataset, batch_size=args.batch_size)
        train.sembert_valid(test_iter, model, args)
        logging.info("END ...")
    else:
        model = ESIM(args).cuda()
        train.sembert_train(train_iter, valid_iter, model, args)
        logging.info("END ...")


def save_data(features, path, itos):
    sen1 = np.array([feature.inputs[0] for feature in features])
    sen2 = np.array([feature.inputs[1] for feature in features])
    sen1_mask = np.array([feature.inputs_mask[0] for feature in features])
    sen2_mask = np.array([feature.inputs_mask[1] for feature in features])
    sen1_tag = np.array([feature.inputs_tag_ids[0] for feature in features])
    sen2_tag = np.array([feature.inputs_tag_ids[1] for feature in features])
    sen1_start_end = np.array([feature.inputs_word_start_end[0] for feature in features])
    sen2_start_end = np.array([feature.inputs_word_start_end[1] for feature in features])
    label = np.array([feature.label_ids for feature in features])
    itos = np.array(itos)
    np.savez(path, sen1=sen1, sen2=sen2, sen1_mask=sen1_mask, sen2_mask=sen2_mask,
             sen1_tag=sen1_tag, sen2_tag=sen2_tag, sen1_start_end=sen1_start_end,
             sen2_start_end=sen2_start_end, label=label, vocab=itos)


def SemBert_data_prepare(args):
    logging.info("Start prepare data...")
    train_data = load_data(args.train_path)
    valid_data = load_data(args.valid_path)
    test_data = load_data(args.test_path)
    train_data = train_data[:20]
    valid_data = valid_data[:20]
    test_data = test_data[:20]
    label_vocab = ({"A2B": 0, "B2A": 1, "Neutral": 2}, ["A2B", "B2A", "Neutral"])
    args.class_num = 3
    threads = []
    for i in range(4):
        thread = MyThread(gen_feature_v2, args=(train_data[int(len(train_data) * (0.25 * i)):int(len(train_data) * (0.25 * (i + 1)))],
                          label_vocab, args))
        threads.append(thread)
    thread_valid = MyThread(gen_feature_v2, args=(valid_data,
                            label_vocab, args))
    thread_test = MyThread(gen_feature_v2, args=(test_data,
                           label_vocab, args))
    for thread in threads:
        thread.start()
    thread_valid.start()
    thread_test.start()
    for thread in threads:
        thread.join()
    thread_valid.join()
    thread_test.join()
    train_features = []
    for thread in threads:
        train_features += thread.get_result()
    logging.info("train_features size:{}".format(len(train_features)))
    # train_features = gen_feature_v2(train_data[:int(len(train_data) * 0.01)], tokenizer, ltp, label_vocab, args)
    # valid_features = gen_feature_v2(valid_data, label_vocab, args)
    valid_features = thread_valid.get_result()
    test_features = thread_test.get_result()
    itos = ['<PAD>', '<CLS>', '<SEP>', 'O', 'A1', 'A0', 'Verb', 'ARGM-ADV', 'ARGM-TMP', 'A2', 'ARGM-DIS', 'ARGM-LOC', 'ARGM-PRP', 'ARGM-MNR', 'ARGM-CND', 'ARGM-DIR', 'ARGM-TPC', 'ARGM-BNF', 'ARGM-EXT', 'A0-CRD', 'A0-PSR', 'A3', 'A0-PSE', 'ARGM-FRQ', 'A0-PRD', 'A1-QTY', 'A1-CRD', 'A1-PSR', 'A1-PSE', 'ARGM-DGR', 'A1-PRD', 'ARGM-PRD', 'A4']
    stoi = {char: idx for idx, char in enumerate(itos)}
    # stoi, itos = build_srl_vocab(train_features)
    args.tag_vocab_size = len(itos)
    logging.info(itos)
    threads = []
    for i in range(4):
        thread = MyThread(transform_srl_feature_v3, args=(train_features[int(len(train_features) * (0.25 * i)):int(len(train_features) * (0.25 * (i + 1)))],
                          stoi, args))
        threads.append(thread)
    thread_valid = MyThread(transform_srl_feature_v3, args=(valid_features,
                            stoi, args))
    thread_test = MyThread(transform_srl_feature_v3, args=(test_features,
                           stoi, args))
    for thread in threads:
        thread.start()
    thread_valid.start()
    thread_test.start()
    for thread in threads:
        thread.join()
    thread_valid.join()
    thread_test.join()
    train_features = []
    for thread in threads:
        train_features += thread.get_result()
    # train_features = transform_srl_feature_v2(train_features, stoi, args)
    # valid_features = transform_srl_feature_v2(valid_features, stoi, args)
    valid_features = thread_valid.get_result()
    test_features = thread_test.get_result()
    logging.info("train_features size:{}".format(len(train_features)))
    logging.info("valid_features size:{}".format(len(valid_features)))
    logging.info("Finished prepare raw data...")
    save_data(train_features, args.train_save_path, itos)
    save_data(valid_features, args.valid_save_path, itos)
    save_data(test_features, args.test_save_path, itos)
    logging.info("END...")


def test(args):
    # train_data = common.load_data(args.train_path)
    # valid_data = common.load_data(args.valid_path)
    # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    # _, label_vocab = common.build_vocab(train_data)
    # _, label_vocab = common.build_vocab(valid_data)
    sen1 = "平凡不意味着无价值，平凡不意味着无作为。"
    sen2 = "螺丝钉看起来很渺小，但它的价值不容我们忽视。少了他，大楼会倒塌。"
    label = "B2A"
    Examples = [Example(sen1, sen2, label)]
    label_vocab = ({"A2B": 0, "B2A": 1, "Neutral": 2}, ["A2B", "B2A", "Neutral"])
    args.class_num = 3
    itos = ['<PAD>', '<CLS>', '<SEP>', 'O', 'A1', 'A0', 'Verb', 'ARGM-ADV', 'ARGM-TMP', 'A2', 'ARGM-DIS', 'ARGM-LOC', 'ARGM-PRP', 'ARGM-MNR', 'ARGM-CND', 'ARGM-DIR', 'ARGM-TPC', 'ARGM-BNF', 'ARGM-EXT', 'A0-CRD', 'A0-PSR', 'A3', 'A0-PSE', 'ARGM-FRQ', 'A0-PRD', 'A1-QTY', 'A1-CRD', 'A1-PSR', 'A1-PSE', 'ARGM-DGR', 'A1-PRD', 'ARGM-PRD', 'A4']
    stoi = {l: idx for idx, l in enumerate(itos)}
    features = gen_feature_v2(Example, label_vocab, args)
    features = transform_srl_feature_v3(features, stoi, args)

    # # # model_name = "bert-base-chinese"
    # # tokenizer = BertTokenizer.from_pretrained(args.model_name)
    # # ltp = LTP()
    # logging.info("Start...")
    # Examples = [Example(sen1, sen2, label), Example(sen1, sen2, label)]
    # label_vocab = ({"A2B": 0, "B2A": 1, "Neutral": 2}, ["A2B", "B2A", "Neutral"])
    # # args.class_num = 3
    # features = gen_feature_v2(Examples, label_vocab, args)
    # srl_stoi, srl_itos = build_srl_vocab(features)
    # # args.tag_vocab_size = len(srl_itos)
    # # logging.info(srl_itos)
    # features = transform_srl_feature_v3(features, srl_stoi, args)
    # for feature in features:
    #     logging.info("inputs:\n{}\n{}".format(feature.inputs[0], feature.inputs[1]))
    #     logging.info("inputs mask:\n{}\n{}".format(feature.inputs_mask[0], feature.inputs_mask[1]))
    #     # logging.info("inputs sen ids:\n{}\n{}".format(feature.inputs_sen_ids))
    #     logging.info("inputs tag ids:\n{}\n{}".format(feature.inputs_tag_ids[0], feature.inputs_tag_ids[1]))
    #     logging.info("label ids:\n{}".format(feature.label_ids))
    #     cnt = 0
    #     tag_cnt = 0
    #     for i in feature.inputs[1]:
    #         if i != 0:
    #             cnt += 1
    #     for i in feature.inputs_tag_ids[1][0]:
    #         if i != 0:
    #             tag_cnt += 1
    #     logging.info("{} : {}".format(cnt, tag_cnt))
    #     assert cnt == tag_cnt
    test_sen1 = torch.LongTensor([feature.inputs[0] for feature in features])
    test_sen2 = torch.LongTensor([feature.inputs[1] for feature in features])
    test_sen1_mask = torch.LongTensor([feature.inputs_mask[0] for feature in features])
    test_sen2_mask = torch.LongTensor([feature.inputs_mask[1] for feature in features])
    test_sen1_tag = torch.LongTensor([feature.inputs_tag_ids[0] for feature in features])
    test_sen2_tag = torch.LongTensor([feature.inputs_tag_ids[1] for feature in features])
    test_sen1_start_end = torch.LongTensor([feature.inputs_word_start_end[0] for feature in features])
    test_sen2_start_end = torch.LongTensor([feature.inputs_word_start_end[1] for feature in features])
    test_label = torch.LongTensor([feature.label_ids for feature in features])
    test_dataset = TensorDataset(test_sen1, test_sen1_mask, test_sen1_tag, test_sen1_start_end,
                                 test_sen2, test_sen2_mask, test_sen2_tag, test_sen2_start_end,
                                 test_label)
    test_iter = DataLoader(test_dataset, batch_size=1)
    model = torch.load(args.load_model_path).cuda()
    model.eval()
    # model = ESIM(args).cuda()
    # model.train()
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    for sen1, sen1_mask, sen1_tag, sen1_start_end, \
        sen2, sen2_mask, sen2_tag, sen2_start_end, label in test_iter:
        logging.info("Start train")
        logging.info(sen1)
        output = model(sen1.cuda(), sen1_mask.cuda(), sen1_tag.cuda(), sen1_start_end.cuda(),
                       sen2.cuda(), sen2_mask.cuda(), sen2_tag.cuda(), sen2_start_end.cuda())
        e = model.e
        logging.info(e)
    #     logging.info("output:{}".format(output.size()))
        # logging.info("inputs word start end:{}".foramt(feature.word_start_end))
    
    # model.train()
    # logit = model(sen1, sen2)
    # loss = torch.nn.functional.cross_entropy(logit.cpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="data/train.json")
    parser.add_argument("--valid-path", type=str, default="data/valid.json")
    parser.add_argument("--test-path", type=str, default="data/test.json")
    parser.add_argument("--model-name", type=str, default="bert-base-chinese")
    parser.add_argument("--bert-name", type=str, default="bert-base-chinese")
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
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--filter-size", type=int, default=2)
    parser.add_argument("--tag-hidden-size", type=int, default=10)
    parser.add_argument("--tag-output-dim", type=int, default=10)
    parser.add_argument("--tag-num-layer", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=500)
    parser.add_argument("--no-tag", action="store_false")
    parser.add_argument("--train-save-path", type=str)
    parser.add_argument("--valid-save-path", type=str)
    parser.add_argument("--test-save-path", type=str)
    parser.add_argument("--train-load-path", type=str)
    parser.add_argument("--valid-load-path", type=str)
    parser.add_argument("--test-load-path", type=str)
    parser.add_argument("--model-save-path", type=str)
    parser.add_argument("--model-load-path", type=str)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    if args.train_type == "bert-only":
        main(args)
    elif args.train_type == "bert-ESIM":
        bert_ESIM_main(args)
    elif args.train_type == "test":
        test(args)
    elif args.train_type == "SemBert-ESIM":
        Sembert_ESIM_main(args)
    elif args.train_type == "SemBert-prepare":
        SemBert_data_prepare(args)
