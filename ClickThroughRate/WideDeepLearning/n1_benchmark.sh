DEVICE_NUM_PER_NODE=1
DATA_ROOT=/dataset/wdl_ofrecord/ofrecord
EMBD_SIZE=2322444
BATHSIZE=2048

python3 wdl_train_eval.py \
  --learning_rate=0.001 \
  --batch_size=$BATHSIZE \
  --train_data_dir $DATA_ROOT/train \
  --loss_print_every_n_iter=100 \
  --eval_interval=0 \
  --deep_dropout_rate=0.5 \
  --max_iter=310 \
  --hidden_units_num=7\
  --hidden_size=1024 \
  --wide_vocab_size=$EMBD_SIZE \
  --deep_vocab_size=$EMBD_SIZE \
  --train_data_part_num 256 \
  --train_part_name_suffix_length=5 \
  --eval_data_dir $DATA_ROOT/val \
  --eval_data_part_num 256 \
  --eval_part_name_suffix_length=5 \
  --gpu_num_per_node $DEVICE_NUM_PER_NODE \
  --num_dataloader_thread_per_gpu 1 \
  --node_ips '127.0.0.1' \
