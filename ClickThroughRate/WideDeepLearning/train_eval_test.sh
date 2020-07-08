rm -rf core.*
export ENABLE_USER_OP=True

#EMBD_SIZE=3200000 
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

