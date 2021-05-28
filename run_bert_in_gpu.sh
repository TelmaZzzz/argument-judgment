#!/bin/bash
source ~/.bashrc

source activate telma

PATH=~/anaconda3/envs/telma/bin:$PATH
python src/bert_main.py \
--train-type="bert-only" \
--batch-size=10 \
--learning-rate=0.00002 \
--train-path="data/three_class_v3/support/train.json" \
--valid-path="data/three_class_v3/support/valid.json" \
--epoch=4 \
--eval-step=500 \
--model-save-path="/users10/lyzhang/model/argument-judgment/SUPPORT_BERT" \
# > train.log 2>&1 &

# python src/bert_main.py \
# --train-type="SemBert-prepare" \
# --train-path="data/three_class_v3/train.json" \
# --valid-path="data/three_class_v3/valid.json" \
# --test-path="data/three_class_v3/test.json" \
# --train-save-path="data/three_class_v3/prepare/train_v3.npz" \
# --valid-save-path="data/three_class_v3/prepare/valid_v3.npz" \
# --test-save-path="data/three_class_v3/prepare/test_v3.npz" \
# --batch-size=16 \
# --learning-rate=0.00002 \
# --fc1-dim=300 \
# --fc2-dim=50 \
# --dropout=0.5 \
# --max-length=100 \