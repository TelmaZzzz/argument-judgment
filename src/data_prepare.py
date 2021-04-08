import os
import sys
import argparse
import random
import json
import importlib
importlib.reload(sys)
import codecs  
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach()) 
sys.path.append(os.getcwd())
sys.path.append("/users10/lyzhang/opt/tiger/argument-judgment")
from util import common

def output(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            # print(item)
            # temp = json.dumps(item, ensure_ascii=False)
            # print(temp)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def shuffle_list(data):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    re_data = [data[i] for i in index]
    return re_data

def main(args):
    raw_data = common.read_raw_data(args.data_path)
    index = [i for i in range(len(raw_data))]
    random.shuffle(index)
    train_index = index[:int(len(index)*0.8)]
    valid_index = index[int(len(index)*0.8):int(len(index)*0.9)]
    test_index = index[int(len(index)*0.9):]
    train_raw_data = {j: raw_data[i] for j, i in enumerate(train_index)}
    valid_raw_data = {j: raw_data[i] for j, i in enumerate(valid_index)}
    test_raw_data = {j: raw_data[i] for j, i in enumerate(test_index)}
    train_data = common.get_idea2support(train_raw_data) + common.get_theis2idea(train_raw_data) + common.get_neutral(train_raw_data, 16000)
    valid_data = common.get_idea2support(valid_raw_data) + common.get_theis2idea(valid_raw_data) + common.get_neutral(valid_raw_data, 2000)
    test_data = common.get_idea2support(test_raw_data) + common.get_theis2idea(test_raw_data) + common.get_neutral(test_raw_data, 2000)
    output(args.train_path, shuffle_list(train_data))
    output(args.valid_path, shuffle_list(valid_data))
    output(args.test_path, shuffle_list(test_data))

def three_class_main(args):
    raw_data = common.read_raw_data(args.data_path)
    index = [i for i in range(len(raw_data))]
    random.shuffle(index)
    train_index = index[:int(len(index)*0.8)]
    valid_index = index[int(len(index)*0.8):int(len(index)*0.9)]
    test_index = index[int(len(index)*0.9):]
    train_raw_data = {j: raw_data[i] for j, i in enumerate(train_index)}
    valid_raw_data = {j: raw_data[i] for j, i in enumerate(valid_index)}
    test_raw_data = {j: raw_data[i] for j, i in enumerate(test_index)}
    train_data = common.get_support(train_raw_data) + common.get_theis_support(train_raw_data) + common.get_neutral(train_raw_data, int(args.none_num*0.8))
    valid_data = common.get_support(valid_raw_data) + common.get_theis_support(valid_raw_data) + common.get_neutral(valid_raw_data, int(args.none_num*0.1))
    test_data = common.get_support(test_raw_data) + common.get_theis_support(test_raw_data) + common.get_neutral(test_raw_data, int(args.none_num*0.1))
    output(args.train_path, shuffle_list(train_data))
    output(args.valid_path, shuffle_list(valid_data))
    output(args.test_path, shuffle_list(test_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/seg_sentence_3.txt")
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--valid-path", type=str)
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--none-num", type=int, default=20000)
    args = parser.parse_args()
    three_class_main(args)