#!/bin/bash
source ~/.bashrc

source activate telma

# PATH=~/anaconda3/envs/telma/bin:$PATH
python src/bert_main.py \
--train-type="bert-ESIM" \
--epoch=60 \
--batch-size=16 \
--learning-rate=0.000001 \
--fc1-dim=300 \
--fc2-dim=50 \
--train-path="data/three_class/train.json" \
--valid-path="data/three_class/valid.json" \
# > train.log 2>&1 &
