#!/bin/bash
source ~/.bashrc

source activate telma

# PATH=~/anaconda3/envs/telma/bin:$PATH
python src/bert_main.py \
--train-type="bert-only"
--batch-size=32 \
--learning-rate=0.00001 \
--train-path="data/three_class/train.json" \
--valid-path="data/three_class/valid.json" \
# > train.log 2>&1 &
