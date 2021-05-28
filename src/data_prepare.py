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
import logging
logging.getLogger().setLevel(logging.INFO)

def output(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            # print(item)
            # temp = json.dumps(item, ensure_ascii=False)
            # print(temp)
            if len(item["sen1"]) <= 15 or len(item["sen2"]) <= 15:
                continue
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

def three_class_main_v2(args):
    raw_data = common.read_raw_data_v2(args.data_path)
    logging.info("raw data finish")
    index = [i for i in range(len(raw_data))]
    random.shuffle(index)
    train_index = index[:int(len(index)*0.8)]
    valid_index = index[int(len(index)*0.8):int(len(index)*0.9)]
    test_index = index[int(len(index)*0.9):]
    train_raw_data = {j: raw_data[i] for j, i in enumerate(train_index)}
    valid_raw_data = {j: raw_data[i] for j, i in enumerate(valid_index)}
    test_raw_data = {j: raw_data[i] for j, i in enumerate(test_index)}
    test_data = common.get_support_v2(test_raw_data) + common.get_theis_support(test_raw_data) + common.get_neutral(test_raw_data, int(args.none_num*0.1))
    logging.info("test finish")
    train_data = common.get_support_v2(train_raw_data) + common.get_theis_support(train_raw_data) + common.get_neutral(train_raw_data, int(args.none_num*0.8))
    logging.info("train finish")
    valid_data = common.get_support_v2(valid_raw_data) + common.get_theis_support(valid_raw_data) + common.get_neutral(valid_raw_data, int(args.none_num*0.1))
    logging.info("valid finish")
    output(args.train_path, shuffle_list(train_data))
    output(args.valid_path, shuffle_list(valid_data))
    output(args.test_path, shuffle_list(test_data))
    logging.info("END")


def gen_data(raw_data):
    support_data = common.get_support_v2(raw_data)
    support_data += common.get_neutral(raw_data, int(len(support_data) * 0.5))
    theis_data = common.get_theis_support(raw_data)
    theis_data += common.get_neutral(raw_data, int(len(theis_data) * 0.5))
    return support_data, theis_data


def three_class_main_v3(args):
    raw_data = common.read_raw_data_v2(args.data_path)
    logging.info("raw data finish")
    index = [i for i in range(len(raw_data))]
    random.shuffle(index)
    train_index = index[:int(len(index)*0.8)]
    valid_index = index[int(len(index)*0.8):int(len(index)*0.9)]
    test_index = index[int(len(index)*0.9):]
    train_raw_data = {j: raw_data[i] for j, i in enumerate(train_index)}
    valid_raw_data = {j: raw_data[i] for j, i in enumerate(valid_index)}
    test_raw_data = {j: raw_data[i] for j, i in enumerate(test_index)}
    train_support_data, train_theis_data = gen_data(train_raw_data)
    valid_support_data, valid_theis_data = gen_data(valid_raw_data)
    test_support_data, test_theis_data = gen_data(test_raw_data)
    output(args.root_path + "/support/train.json", shuffle_list(train_support_data))
    output(args.root_path + "/theis/train.json", shuffle_list(train_theis_data))
    output(args.root_path + "/support/valid.json", shuffle_list(valid_support_data))
    output(args.root_path + "/theis/valid.json", shuffle_list(valid_theis_data))
    output(args.root_path + "/support/test.json", shuffle_list(test_support_data))
    output(args.root_path + "/theis/test.json", shuffle_list(test_theis_data))
    train_data = train_support_data + train_theis_data
    valid_data = valid_support_data + valid_theis_data
    test_data = test_support_data + test_theis_data
    output(args.train_path, shuffle_list(train_data))
    output(args.valid_path, shuffle_list(valid_data))
    output(args.test_path, shuffle_list(test_data))
    logging.info("END")


if __name__ == "__main__":
    logging.info("Start....")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/seg_sentence_3.txt")
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--valid-path", type=str)
    parser.add_argument("--test-path", type=str)
    parser.add_argument("--root-path", type=str)
    parser.add_argument("--none-num", type=int, default=20000)
    parser.add_argument("--mode", type=str, default="three class")
    args = parser.parse_args()
    if args.mode == "three class":
        three_class_main(args)
    elif args.mode == "three class v2":
        three_class_main_v3(args)
    elif args.mode == "main":
        main(args)