python src/data_prepare.py \
--train-path="data/three_class_v2/train.json" \
--valid-path="data/three_class_v2/valid.json" \
--test-path="data/three_class_v2/test.json" \
--none-num=50000 \
--mode="three class v2" \
> data_prepare.log 2>&1 &