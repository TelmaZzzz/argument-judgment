import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def eval(valid_iter, model, args):
    model.eval()
    acc = 0
    total = 0
    for sen1, sen2, sen1_mask, sen2_mask, label in train_iter:
        logit = model(sen1, sen2)
        acc += (torch.max(logit, 1)[1].view(label.size()) == label).sum()
        total += label.size()[0]
    logging.info("auc: {:.4f}%({}/{})".format(acc / total) * 100, acc, total)

def train(train_iter, valid_iter, model, args):
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    for step in range(args.epoch):
        model.train()
        for sen1, sen2, sen1_mask, sen2_mask, label in train_iter:
            logit = model(sen1, sen2)
            loss = F.cross_entropy(logit, label.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eval(valid_iter, model, args)
            
            
    

