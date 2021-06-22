#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

gpt2_dir=gpt-2
gpt2_path=/OneFlow-Benchmark/LanguageModeling/$gpt2_dir
model_path=/wksp/gpt2_small_of_models_4_regression
dataset=$model_path/data/wiki_00
cfg_dir=$model_path/117M

gpu_num_per_node=$1
batch_size=8
n_embd=1536
n_head=16
n_layer=12
seq_len=1024
dropout_rate=0.1

log_dir=log
test_case=bsz${batch_size}_g${gpu_num_per_node}_e${n_embd}_h${n_head}_l${n_layer}
mem_file=$log_dir/$test_case.mem
log_file=$log_dir/$test_case.log

cmd=""
# cmd+='gdb --arg "
# cmd+="/opt/nvidia/nsight-systems-2020.5.1/bin/nsys profile -o $test_case "
cmd+="python3 -m $gpt2_dir.src.train "
cmd+="--dataset=$dataset "
cmd+="--cfg_dir=$cfg_dir "
cmd+="--n_vocab=50272 "
cmd+="--n_ctx=1024 "
cmd+="--n_embd=$n_embd "
cmd+="--n_head=$n_head "
cmd+="--n_layer=$n_layer "
cmd+="--seq_len=$seq_len "
cmd+="--embedding_dropout=$dropout_rate "
cmd+="--hidden_dropout=$dropout_rate "
cmd+="--attention_dropout=$dropout_rate "
cmd+="--use_fp16=True "
cmd+="--use_big_fc=True "
cmd+="--checkpoint-activations "
cmd+="--optimizer=adam "
cmd+="--parallel-embedding "
cmd+="--parallel-decoder "
cmd+="--parallel-loss "
cmd+="--iter_num=110 "
cmd+="--loss_print_every_n_iter=10 "
cmd+="--metric-print-format=table "
cmd+="--total_batch_size=$batch_size "
cmd+="--gpu_num_per_node=$gpu_num_per_node "
cmd+="--num_nodes=1 "
cmd+="--node_ips=10.11.0.2,10.11.0.3 "

set -x

# rm -rf core.* output
mkdir -p $log_dir
python3 $gpt2_path/tools/gpu_memory_usage.py \
  -g $gpu_num_per_node \
  -n 0.5 \
  > $mem_file 2>&1 </dev/null &
ln -s $gpt2_path
$cmd| tee $log_file
