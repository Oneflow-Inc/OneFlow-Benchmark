The main different between `wdl_train_eval.py` and `wdl_train_eval_test.py` is:
`wdl_train_eval_test.py` is a end to end process of n-epoch training with training dataset, evaluation with full eval dataset after training of every epoch and testing with test dataset at the last stage. The main training loop is `epoch`.

Otherwise, in `wdl_train_eval.py`, the main training loop is `iteration`. Only evaluate 20 samples a time but not full eval dataset. and no test stage.

## Run OneFlow-WDL with train and evaluation
```
EMBD_SIZE=1603616 
DATA_ROOT=/DATA/disk1/criteo_wdl/ofrecord
python3 wdl_train_eval.py \
  --train_data_dir $DATA_ROOT/train \
  --train_data_part_num 256 \
  --train_part_name_suffix_length=5 \
  --eval_data_dir $DATA_ROOT/val \
  --eval_data_part_num 256 \
  --max_iter=300000 \
  --loss_print_every_n_iter=1000 \
  --eval_interval=1000 \
  --batch_size=512 \
  --wide_vocab_size=$EMBD_SIZE \
  --deep_vocab_size=$EMBD_SIZE \
  --gpu_num 1
```

## Run OneFlow-WDL with train, evaluation and test 
```
EMBD_SIZE=1603616 
DATA_ROOT=/DATA/disk1/criteo_wdl/ofrecord
python3 wdl_train_eval_test.py \
  --train_data_dir $DATA_ROOT/train \
  --train_data_part_num 256 \
  --train_part_name_suffix_length=5 \
  --eval_data_dir $DATA_ROOT/val \
  --eval_data_part_num 256 \
  --eval_part_name_suffix_length=5 \
  --test_data_dir $DATA_ROOT/test \
  --test_data_part_num 256 \
  --test_part_name_suffix_length=5 \
  --loss_print_every_n_iter=1000 \
  --batch_size=16484 \
  --wide_vocab_size=$EMBD_SIZE \
  --deep_vocab_size=$EMBD_SIZE \
  --gpu_num 1
```

OneFlow-WDL网络实现了模型并行与稀疏更新，在8卡12G TitanV的服务器上实现支持超过4亿的词表大小，而且性能没有损失与小词表性能相当，详细请参考[这篇文档](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/cn/docs/adv_examples/wide_deep.md)评测部分的内容。