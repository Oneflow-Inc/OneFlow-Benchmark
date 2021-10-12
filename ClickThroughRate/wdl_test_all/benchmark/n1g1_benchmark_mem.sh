EMBD_SIZE=1603616 
DATA_ROOT=/dataset/wdl_ofrecord/ofrecord
python3 wdl_train_eval.py \
  --train_data_dir $DATA_ROOT/train \
  --train_data_part_num 256 \
  --train_part_name_suffix_length=5 \
  --eval_data_dir $DATA_ROOT/val \
  --eval_data_part_num 256 \
  --eval_part_name_suffix_length=5 \
  --max_iter=103 \
  --loss_print_every_n_iter=1 \
  --batch_size=16384 \
  --deep_dropout_rate=0.5\
  --hidden_units_num 7\
  --hidden_size 1024\
  --wide_vocab_size=$EMBD_SIZE \
  --deep_vocab_size=$EMBD_SIZE \
  --gpu_num_per_node 1 \
  --test_name 'n1g1_benchmark_mem'