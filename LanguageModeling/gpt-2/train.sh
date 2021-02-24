#!/bin/bash
export PYTHONUNBUFFERED=1

script_path=$(realpath $0)
gpt_root=$(dirname $script_path)
data_path=$gpt_root/data/wiki_00
cfg_path=$gpt_root/models/117M

gpu_num_per_node=8
batch_size=8

n_vocab=50257
n_ctx=1024
seq_len=1024
n_head=12
n_embd=768
n_layer=12
dropout_rate=0.0

# rm -rf core.*
log_dir=log
test_case=bsz${batch_size}_g${gpu_num_per_node}_e${n_embd}_h${n_head}_l${n_layer}
log_file=$log_dir/$test_case.log


mkdir -p $log_dir

python3 -m src.train \
  --dataset=$data_path \
  --cfg_dir=$cfg_path \
  --save_last_snapshot=False \
  --iter_num=100 \
  --loss_print_every_n_iter=1 \
  --total_batch_size=$batch_size \
  --gpu_num_per_node=$gpu_num_per_node \
  --seq_len=$seq_len \
  --n_vocab=$n_vocab \
  --n_ctx=$n_ctx \
  --n_embd=$n_embd \
  --n_head=$n_head \
  --n_layer=$n_layer \
  --optimizer=adam \
  --embedding_dropout=$dropout_rate \
  --hidden_dropout=$dropout_rate \
  --attention_dropout=$dropout_rate \
  --make-vocab-size-divisible-by=0 \
  | tee $log_file
