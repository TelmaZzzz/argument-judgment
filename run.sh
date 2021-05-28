python src/main.py \
--train-path="data/three_class_v3/train.json" \
--valid-path="data/three_class_v3/valid.json" \
--test-path="data/three_class_v3/test.json" \
--model-load-path="/users10/lyzhang/model/argument-judgment/ALL_ESIM.pkl" \
--mode="predict" \
> train.log 2>&1 &