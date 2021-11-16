EMBD_SIZE=1603616 
DATA_ROOT=/dataset/wdl_ofrecord/ofrecord
python3 wdl_train_eval.py \
  --learning_rate=0.001 \
  --batch_size=32 \
  --train_data_dir $DATA_ROOT/train \
  --loss_print_every_n_iter=100 \
  --deep_dropout_rate=0\
  --max_iter=1100 \
  --hidden_units_num=7\
  --hidden_size=1024 \
  --wide_vocab_size=$EMBD_SIZE \
  --deep_vocab_size=$EMBD_SIZE \
  --train_data_part_num 2 \
  --train_part_name_suffix_length=5 \
  --eval_data_dir $DATA_ROOT/val \
  --eval_data_part_num 256 \
  --eval_part_name_suffix_length=5 \
  --gpu_num_per_node 2 \
  --num_dataloader_thread_per_gpu 1 \
  --test_name 'n1g1_benchmark'