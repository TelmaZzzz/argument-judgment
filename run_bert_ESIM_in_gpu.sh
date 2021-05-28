#!/bin/bash
source ~/.bashrc

source activate telma

# PATH=~/anaconda3/envs/telma/bin:$PATH
# python src/bert_main.py \
# --train-type="bert-ESIM" \
# --epoch=20 \
# --batch-size=16 \
# --learning-rate=0.000003 \
# --fc1-dim=300 \
# --fc2-dim=50 \
# --train-path="data/three_class_v3/theis/train.json" \
# --valid-path="data/three_class_v3/theis/valid.json" \
# --eval-step=500 \
# --model-save-path="/users10/lyzhang/model/argument-judgment/THEIS_BERT_ESIM" \
# > train.log 2>&1 &


python src/bert_main.py \
--train-type="SemBert-ESIM" \
--epoch=15 \
--batch-size=16 \
--learning-rate=0.000003 \
--fc1-dim=300 \
--fc2-dim=50 \
--train-path="data/three_class_v3/train.json" \
--valid-path="data/three_class_v3/valid.json" \
--dropout=0.3 \
--eval-step=500 \
--max-length=100 \
--train-load-path="data/three_class_v3/prepare/theis/train_v3.npz" \
--valid-load-path="data/three_class_v3/prepare/theis/valid_v3.npz" \
--tag-output-dim=50 \
--model-save-path="/users10/lyzhang/model/argument-judgment/THEIS_SemBERT_ESIM_3" \