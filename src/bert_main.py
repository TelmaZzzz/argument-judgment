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
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")


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


def test(args):
    sen1 = "在这里，神，并非遥不可及，并非高高在上，而是以一种谦逊的姿态，与身边的一切友好相处。"
    sen2 = "这样你既不会妄自菲薄，也不会妄自尊大，做到谦逊成熟，不断进取，成功便不招自来。"
    label = "ThesisAndIdea"
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = bert_ESIM.BERT_ESIM(args).cuda()
    sen1 = torch.LongTensor([tokenizer.encode(sen1, add_special_tokens=True)])
    sen2 = torch.LongTensor([tokenizer.encode(sen2, add_special_tokens=True)])
    
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
    args = parser.parse_args()
    if args.train_type == "bert-only":
        main(args)
    elif args.train_type == "bert-ESIM":
        bert_ESIM_main(args)
    elif args.train_type == "test":
        test(args)