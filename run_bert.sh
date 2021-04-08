python src/bert_main.py \
--train-type="bert-ESIM" \
--train-path="data/three_class/valid.json" \
--valid-path="data/three_class/test.json" \
> train.log 2>&1 &