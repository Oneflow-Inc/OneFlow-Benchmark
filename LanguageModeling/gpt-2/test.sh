#!/bin/bash
rm -rf core.*
export PYTHONUNBUFFERED=1

dataset=data/wiki_00
cfg_dir=models/117M

gpu_num_per_node=$1
batch_size=4
n_embd=768
n_head=12
n_layer=1
seq_len=1024
dropout_rate=0.1

cmd=""
#cmd+="gdb --args "
# cmd+="/opt/nvidia/nsight-systems-2020.5.1/bin/nsys profile -o $test_case "
cmd+="python3 -m src.train "
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
cmd+="--use_fp16=False "
cmd+="--use_big_fc=False "
cmd+="--checkpoint-activations "
cmd+="--optimizer=sgd "
cmd+="--parallel-loss "
cmd+="--iter_num=110 "
cmd+="--loss_print_every_n_iter=10 "
cmd+="--metric-print-format=table "
cmd+="--total_batch_size=$batch_size "
cmd+="--gpu_num_per_node=$gpu_num_per_node "
cmd+="--embd_model_parallel_size=1 "
cmd+="--attn_model_parallel_size=2 "
cmd+="--num_nodes=1 "
cmd+="--node_ips=10.11.0.2,10.11.0.3 "

set -x

$cmd
