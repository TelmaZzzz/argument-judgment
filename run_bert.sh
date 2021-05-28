# python src/bert_main.py \
# --train-type="SemBert-ESIM" \
# --train-path="data/three_class_v3/train.json" \
# --valid-path="data/three_class_v3/valid.json" \
# --batch-size=8 \
# --learning-rate=0.00002 \
# --fc1-dim=300 \
# --fc2-dim=50 \
# --dropout=0.1 \
# --max-length=100 \
# --eval-step=500 \
# --train-load-path="data/three_class_v3/prepare/theis/train.npz" \
# --valid-load-path="data/three_class_v3/prepare/theis/valid.npz" \
# # --tag-output-dim=0 \
# > train.log 2>&1 &



# python src/bert_main.py \
# --train-type="test" \
# --train-path="data/three_class_v3/support/train.json" \
# --valid-path="data/three_class_v3/support/valid.json" \
# --train-save-path="data/three_class_v3/prepare/support/train.npz" \
# --valid-save-path="data/three_class_v3/prepare/support/valid.npz" \
# --batch-size=16 \
# --learning-rate=0.00002 \
# --fc1-dim=300 \
# --fc2-dim=50 \
# --dropout=0.5 \
# --max-length=100 \
# > train.log 2>&1 &

# python src/bert_main.py \
# --train-type="bert-only" \
# --train-path="data/three_class_v3/train.json" \
# --valid-path="data/three_class_v3/valid.json" \
# --test-path="data/three_class_v3/test.json" \
# --batch-size=16 \
# --learning-rate=0.00002 \
# --fc1-dim=300 \
# --fc2-dim=50 \
# --dropout=0.5 \
# --max-length=100 \
# --model-load-path="/users10/lyzhang/model/argument-judgment/ALL_BERT.pkl" \
# --mode="predict" \
# > train.log 2>&1 &

# python src/bert_main.py \
# --train-type="bert-ESIM" \
# --epoch=20 \
# --batch-size=16 \
# --learning-rate=0.000003 \
# --fc1-dim=300 \
# --fc2-dim=50 \
# --train-path="data/three_class_v3/train.json" \
# --valid-path="data/three_class_v3/valid.json" \
# --test-path="data/three_class_v3/test.json" \
# --eval-step=500 \
# --model-load-path="/users10/lyzhang/model/argument-judgment/ALL_BERT_ESIM.pkl" \
# --mode="predict" \
# > train.log 2>&1 &

python src/bert_main.py \
--train-type="SemBert-ESIM" \
--epoch=15 \
--batch-size=16 \
--learning-rate=0.000003 \
--fc1-dim=300 \
--fc2-dim=50 \
--train-path="data/three_class_v3/theis/train.json" \
--valid-path="data/three_class_v3/theis/valid.json" \
--dropout=0.5 \
--eval-step=500 \
--max-length=100 \
--train-load-path="data/three_class_v3/prepare/train_v3.npz" \
--valid-load-path="data/three_class_v3/prepare/valid_v3.npz" \
--test-load-path="data/three_class_v3/prepare/test_v3.npz" \
--tag-output-dim=50 \
--model-load-path="/users10/lyzhang/model/argument-judgment/ALL_SemBERT_ESIM_3.pkl" \
--mode="predict" \
> train.log 2>&1 &