import torch
import sys
sys.path.append("/home/telma/opt/tiger/argument-judgment")
from util import common
from model import ESIM
from config.config import args
import train
import random
import warnings
import logging
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")

def prepare_data(path):
    raw_data = common.read_raw_data(path)
    data = common.get_theis2idea(raw_data) + common.get_idea2support(raw_data) + common.get_neutral(raw_data, 20000)
    # print(len(data))
    index = [i for i in range(len(data))]
    random.shuffle(index)
    # print(index[:10])
    re_data = [data[i] for i in index]
    # print(len(re_data))
    train_data = re_data[:int(len(re_data)*0.8)]
    valid_data = re_data[int(len(re_data)*0.8):int(len(re_data)*0.9)]
    test_data = re_data[int(len(re_data)*0.9):]
    return train_data, valid_data, test_data

if __name__ == "__main__":
    logging.info("Prepare raw data...")
    train_rawdata, valid_rawdata, test_rawdata = prepare_data(args.data_path)
    # print(len(train_rawdata), len(valid_rawdata), len(test_rawdata))
    sen_vocab, label_vocab = common.build_vocab(train_rawdata)
    train_data = common.word2idx(train_rawdata, sen_vocab, label_vocab)
    valid_data = common.word2idx(valid_rawdata, sen_vocab, label_vocab)
    train_data = common.fix_dataset(train_data, sen_vocab, args.fix_length)
    valid_data = common.fix_dataset(valid_data, sen_vocab, args.fix_length)
    train_sen1, train_sen1_mask, train_sen2, train_sen2_mask, train_label = \
        common.gen_tensor(train_data)
    valid_sen1, valid_sen1_mask, valid_sen2, valid_sen2_mask, valid_label = \
        common.gen_tensor(valid_data)
    train_dataset = torch.utils.data.TensorDataset(train_sen1, train_sen2, train_sen1_mask,
        train_sen2_mask, train_label)
    valid_dataset = torch.utils.data.TensorDataset(valid_sen1, valid_sen2, valid_sen1_mask,
        valid_sen2_mask, valid_label)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
    logging.info("Data preparing finished!!!")
    logging.debug(label_vocab[0])
    args.class_num = len(label_vocab[0])
    args.embed_num = len(sen_vocab[0])
    model = ESIM.ESIM(args)
    train.train(train_iter, valid_iter, model, args)