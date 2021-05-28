python src/data_prepare.py \
--train-path="data/three_class_v3/train.json" \
--valid-path="data/three_class_v3/valid.json" \
--test-path="data/three_class_v3/test.json" \
--root-path="data/three_class_v3" \
--none-num=50000 \
--mode="three class v2" \
> data_prepare.log 2>&1 &