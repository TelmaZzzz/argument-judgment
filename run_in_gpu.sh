#!/bin/bash
source ~/.bashrc

source activate telma

# PATH=~/anaconda3/envs/telma/bin:$PATH
python src/main.py \
--train-path="data/three_class_v3/theis/train.json" \
--valid-path="data/three_class_v3/theis/valid.json" \
--model-save-path="/users10/lyzhang/model/argument-judgment/theis_ESIM" \
# > train.log 2>&1 &
