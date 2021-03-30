import torch
import os
import re
import random

def read_raw_data(url):
    data = dict()
    with open(url, "r", encoding="utf-8") as f:
        raw_data = f.readlines()
    # for index in range(len(raw_data)):
        # raw_data[index] = re.split(r"\t", raw_data[index])
    for line in raw_data:
        line = re.split(r"\t", line)
        if(int(line[0])>=6000): break
        if data.get(int(line[0]), None) is None:
            data[int(line[0])] = dict()
        if data[int(line[0])].get(int(line[1]), None) is None:
            data[int(line[0])][int(line[1])] = dict()
        if data[int(line[0])][int(line[1])].get(line[4], None) is None:
            data[int(line[0])][int(line[1])][line[4]] = []
        if len(line[5])<=3: continue
        data[int(line[0])][int(line[1])][line[4]].append(line[5].strip().replace("|", ""))
    return data

def get_theis2idea(raw_data):
    result = []
    for _, page in raw_data.items():
        thesises = []
        ideas = []
        for _, block in page.items():
            thesises.extend(block.get("thesisSen", []))
            ideas.extend(block.get("ideaSen", []))
        for thesis in thesises:
            for idea in ideas:
                result.append({"sen1": thesis, "sen2": idea, "label": "ThesisAndIdea"})
        for i in range(len(ideas)):
            for j in range(i+1, len(ideas)):
                result.append({"sen1": ideas[i], "sen2": ideas[j], "label": "SameIdea"})
    return result    

def get_idea2support(raw_data):
    result = []
    for _, page in raw_data.items():
        for _, block in page.items():
            ideas = block.get("ideaSen", [])
            supports = block.get("ideasupportSen", [])
            examples = block.get("exampleSen", [])
            for idea in ideas:
                for example in examples:
                    result.append({"sen1": idea, "sen2": example, "label": "ExampleSupport"})
                for support in supports:
                    result.append({"sen1": idea, "sen2": support, "label": "Support"})
            for k in range(len(examples)):
                for p in range(k+1, len(examples)):
                    result.append({"sen1": examples[k], "sen2": examples[p], "label": "SameExample"})
            for k in range(len(supports)):
                for p in range(k+1, len(supports)):
                    result.append({"sen1": supports[k], "sen2": supports[p], "label": "SameSupport"})
    return result

def get_neutral(raw_data, time_step):
    result = []
    mp = {0: "ideaSen", 1: "ideasupportSen", 2: "thsisSen", 3: "exampleSen"}
    while time_step:
        time_step -= 1
        sens1 = []
        sens2 = []
        a = random.randint(0, len(raw_data)-1)
        b = random.randint(0, len(raw_data)-1)
        while a == b:
            a = random.randint(0, len(raw_data)-1)
            b = random.randint(0, len(raw_data)-1)
        for _, block in raw_data[a].items():
            for i in range(4):
                sens1.extend(block.get(mp[i], [])) 
        for _, block in raw_data[b].items():
            for i in range(4):
                sens2.extend(block.get(mp[i], []))
        try:
            result.append({"sen1": sens1[random.randint(0, len(sens1) - 1)], "sen2": sens2[random.randint(0, len(sens2) - 1)], "label": "Neutral"})
        except:
            time_step += 1
    return result
        
def load_vocab(url):
    stoi=dict()
    with open(url, "r", encoding="utf-8") as f:
        itos = f.readlines()
        for i in range(len(itos)):
            itos[i] = itos[i].strip()
            stoi[itos[i]]=i
    return (itos, stoi)

def build_vocab(dataset):
    word_times = dict()
    label_times = dict()
    for data in dataset:
        for word in data["sen1"]:
            if word_times.get(word, None) is None:
                word_times[word] = 1
            else: word_times[word] += 1
        for word in data["sen2"]:
            if word_times.get(word, None) is None:
                word_times[word] = 1
            else: word_times[word] += 1
        if label_times.get(data["label"], None) is None:
            label_times[data["label"]] = 1
        else: label_times[data["label"]] += 1 
    sen_itos = sorted(word_times, key=lambda kv:word_times[kv], reverse=True)
    sen_itos.insert(0, "<UKN>")
    sen_itos.insert(0, "<PAD>")
    sen_stoi = dict()
    label_stoi = dict()
    # print(label_times)
    for index, word in enumerate(sen_itos):
        sen_stoi[word]=index
    label_itos = sorted(label_times, key=lambda kv:label_times[kv], reverse=True)
    for index, word in enumerate(label_itos):
        label_stoi[word]=index
    return (sen_itos, sen_stoi), (label_itos, label_stoi)
        
def word2idx(dataset, word_vocab, label_vocab):
    itos = word_vocab[0]
    stoi = word_vocab[1]
    label_itos = label_vocab[0]
    label_stoi = label_vocab[1]
    new_dataset = []
    for data in dataset:
        sen1 = []
        sen2 = []
        for word in data["sen1"]:
            if stoi.get(word, None) is None:
                sen1.append(stoi["<UKN>"])
            else : sen1.append(stoi[word])
        for word in data["sen2"]:
            if stoi.get(word, None) is None:
                sen2.append(stoi["<UKN>"])
            else : sen2.append(stoi[word])
        new_dataset.append({"sen1": sen1, "sen2": sen2, "label": [label_stoi[data["label"]]]})
    return new_dataset

def fix_dataset(dataset, word_vocab, fix_length):
    itos = word_vocab[0]
    stoi = word_vocab[1]
    for i in range(len(dataset)):
        dataset[i]["sen1_mask"] = ([0] * len(dataset[i]["sen1"]) + [1] * max(0, fix_length - len(dataset[i]["sen1"])))[:fix_length]
        dataset[i]["sen1"] = (dataset[i]["sen1"] + [stoi["<PAD>"]] * max(0, fix_length - len(dataset[i]["sen1"])))[:fix_length]
        dataset[i]["sen2_mask"] = ([0] * len(dataset[i]["sen2"]) + [1] * max(0, fix_length - len(dataset[i]["sen2"])))[:fix_length]
        dataset[i]["sen2"] = (dataset[i]["sen2"] + [stoi["<PAD>"]] * max(0, fix_length - len(dataset[i]["sen2"])))[:fix_length]   
    return dataset

def gen_tensor(dataset):
    sen1 = torch.LongTensor([item["sen1"] for item in dataset])
    sen1_mask = torch.LongTensor([item["sen1_mask"] for item in dataset])
    sen2 = torch.LongTensor([item["sen2"] for item in dataset])
    sen2_mask = torch.LongTensor([item["sen2_mask"] for item in dataset])
    label = torch.LongTensor([item["label"] for item in dataset])
    return sen1, sen1_mask, sen2, sen2_mask, label
# def prepare_dataset(train_data, valid_data, sen_vocab, label_vocab):
    # train_dataset = []
    # for data in train_data:
        
    
    

if __name__ == "__main__":
    itos, stoi = load_vocab("./data/bert/vocab.txt")
    raw_data = read_raw_data("./data/seg_sentence_3.txt")
    # print(raw_data[0])
    results = get_neutral(raw_data, 20000) + get_idea2support(raw_data) + get_theis2idea(raw_data)
    print(len(results))
    # results = get_idea2support(raw_data)
    # results = get_theis2idea(raw_data)
    # print(results[:5])
    word_vocab, label_vocab = build_vocab(results)
    print(label_vocab[0])
    # print(word_vocab[0])
    # print(word_vocab[1])
    # dataset = word2idx(results[6:10], word_vocab, label_vocab)
    # print(dataset[0])
    # dataset = fix_dataset(dataset, word_vocab, 40)
    # print(dataset[0])
    # cnt1, cnt2 = 0, 0
    # for result in results:
    #     if result.get("lable", "") == "ExampleSupport":
    #         cnt1 += 1
    #     elif result.get("lable", "") == "SameExample":
    #         cnt2 += 1
    # print(cnt1, cnt2)
    # print(len(results))
    