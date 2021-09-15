echo "数据集构建" \
> yanshi.log 2>&1 && \


python src/data_prepare.py \
--train-path="data/yanshi/train.json" \
--valid-path="data/yanshi/valid.json" \
--test-path="data/yanshi/test.json" \
--root-path="data/yanshi" \
--none-num=50000 \
--mode="three class v2" \
>> yanshi.log 2>&1 && \


echo "数据预处理" \
>> yanshi.log 2>&1 && \


python src/bert_main.py \
--train-type="SemBert-prepare" \
--train-path="data/three_class_v3/train.json" \
--valid-path="data/three_class_v3/valid.json" \
--test-path="data/three_class_v3/test.json" \
--train-save-path="data/yanshi/prepare/train_v3.npz" \
--valid-save-path="data/yanshi/prepare/valid_v3.npz" \
--test-save-path="data/yanshi/prepare/test_v3.npz" \
--batch-size=16 \
--learning-rate=0.00002 \
--fc1-dim=300 \
--fc2-dim=50 \
--dropout=0.5 \
--max-length=100 \
>> yanshi.log 2>&1 && \


echo "模型测试" \
>> yanshi.log 2>&1 && \


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
--train-load-path="data/three_class_v3/prepare/support/train_v3.npz" \
--valid-load-path="data/three_class_v3/prepare/support/valid_v3.npz" \
--test-load-path="data/three_class_v3/prepare/support/test_v3.npz" \
--tag-output-dim=50 \
--model-load-path="/users10/lyzhang/model/argument-judgment/SUPPORT_SemBERT_ESIM_3.pkl" \
--mode="predict" \
>> yanshi.log 2>&1 && \


echo "模型训练" \
>> yanshi.log 2>&1 && \


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
--train-load-path="data/three_class_v3/prepare/support/train_v3.npz" \
--valid-load-path="data/three_class_v3/prepare/support/valid_v3.npz" \
--tag-output-dim=50 \
--model-save-path="/users10/lyzhang/model/argument-judgment/YANSHI" \
>> yanshi.log 2>&1 & \
