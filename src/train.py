import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def eval(valid_iter, model, args):
    logging.info("Start eval...")
    model.eval()
    acc = 0
    total = 0
    for sen1, sen2, sen1_mask, sen2_mask, label in valid_iter:
        logit = model(sen1.cuda(), sen2.cuda())
        label = label.squeeze(-1)
        acc += (torch.max(logit.cpu(), 1)[1].view(label.size()) == label).sum()
        total += label.size()[0]
    logging.info("acc: {:.4f}%({}/{})".format((acc / total) * 100, acc, total))
    logging.info("Finished eval!!!")

def train(train_iter, valid_iter, model, args):
    logging.info("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    for step in range(args.epoch):
        loss_sum = 0
        model.train()
        for sen1, sen2, sen1_mask, sen2_mask, label in train_iter:
            logit = model(sen1.cuda(), sen2.cuda())
            loss = F.cross_entropy(logit.cpu(), label.squeeze(-1))
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info("loss:{:.4f}".format(loss_sum / len(train_iter)))
        with torch.no_grad():
            eval(valid_iter, model, args)
    logging.info("Finished training!!!") 

def bert_eval(valid_iter, model, args):
    logging.info("Start Eval...")
    model.eval()
    acc = 0
    total = 0
    for sen, label in valid_iter:
        logit = model(sen.cuda())
        label = label.squeeze(-1)
        acc += (torch.max(logit.cpu(), 1)[1].view(label.size()) == label).sum()
        total += label.size()[0]
    logging.info("acc: {:.4f}%({}/{})".format((acc / total) * 100, acc, total))
    logging.info("Finished eval!!!")

def bert_train(train_iter, valid_iter, model, args):
    logging.info("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    for step in range(args.epoch):
        loss_sum = 0
        model.train()
        for sen, label in train_iter:
            logit = model(sen.cuda())
            loss = F.cross_entropy(logit.cpu(), label.squeeze(-1))
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info("loss:{:.4f}".format(loss_sum / len(train_iter)))
        with torch.no_grad():
            bert_eval(valid_iter, model, args)
    logging.info("Finished Training!!!")

def two_sen_eval(valid_iter, model, args):
    logging.info("Starting eval...")
    model.eval()
    acc = 0
    total = 0
    for sen1, sen2, label in valid_iter:
        logit = model(sen1.cuda(), sen2.cuda())
        # label = label.squeeze(-1)
        acc += (torch.max(logit.cpu(), 1)[1].view(label.size()) == label).sum()
        total += label.size()[0]
    logging.info("acc: {:.4f}%({}/{})".format((acc / total) * 100, acc, total))
    logging.info("Finished eval!!!")

def two_sen_train(train_iter, valid_iter, model, args):
    logging.info("Starting training...")
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    for step in range(args.epoch):
        loss_sum = 0
        model.train()
        acc = 0
        total = 0
        for sen1, sen2, label in train_iter:
            logit = model(sen1.cuda(), sen2.cuda())
            logging.debug("label size:{}".format(label.size()))
            logging.debug("logit size:{}".format(logit.cpu().size()))
            loss = F.cross_entropy(logit.cpu(), label)
            acc += (torch.max(logit.cpu(), 1)[1].view(label.size()) == label).sum()
            total += label.size()[0]
            loss_sum += loss.item()
            # logging.info("loss:{:.4f}".format(loss.item()))
            # logging.info("logit:{}".format(logit.cpu()))
            # logging.info("label:{}".format(label))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info("train loss: {:.4f}".format(loss_sum / len(train_iter)))
        logging.info("train acc: {:.4f}%({}/{})".format((acc / total) * 100, acc, total))
        with torch.no_grad():
            two_sen_eval(valid_iter, model, args)
    logging.info("Finished Training!!!")