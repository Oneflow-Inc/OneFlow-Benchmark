#!/bin/bash
export PYTHONUNBUFFERED=1

gpt2_dir=gpt-2
gpt2_path=/OneFlow-Benchmark/LanguageModeling/$gpt2_dir
model_path=/wksp/gpt2_small_of_models_4_regression
data_path=$model_path/data/wiki_00
cfg_path=$model_path/117M
model_load_path=$model_path/of_models_new

gpu_num_per_node=1
batch_size=1

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

set -x

mkdir -p $log_dir
ln -s $gpt2_path

python3 -m $gpt2_dir.src.train \
  --dataset=$data_path \
  --cfg_dir=$cfg_path \
  --model_load_dir=$model_load_path \
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
