#!/bin/bash

export PYTHONUNBUFFERED=1

TASK="LAMBADA"
VALID_DATA=/path/to/lambada_test.json
VOCAB_FILE=/path/to/gpt2-vocab.json
MERGE_FILE=/path/to/gpt2-merges.txt
CHECKPOINT_PATH=/path/to/model


gpu_num_per_node=1
micro_batch_size=8
hidden_size=768
num_attn_heads=12
num_layers=12
seq_length=1024
dropout_rate=0.0

cmd=""
cmd+="python3 tasks/main.py "
cmd+="--task $TASK "
cmd+="--valid-data $VALID_DATA "
cmd+="--tokenizer-type GPT2BPETokenizer "
cmd+="--strict-lambada "
cmd+="--merge-file $MERGE_FILE "
cmd+="--vocab-file $VOCAB_FILE "
cmd+="--load $CHECKPOINT_PATH "
cmd+="--dataset $VALID_DATA "
cmd+="--vocab-size 50257 "
cmd+="--hidden-size $hidden_size "
cmd+="--num-attention-heads $num_attn_heads "
cmd+="--num-layers $num_layers "
cmd+="--seq-length $seq_length "
cmd+="--hidden-dropout $dropout_rate "
cmd+="--attention-dropout $dropout_rate "
cmd+="--fp16 "
cmd+="--checkpoint-activations "
cmd+="--multihead-attention-fusion "
cmd+="--make-vocab-size-divisible-by=128 "
cmd+="--log-interval=10 "
cmd+="--metric-print-format=table "
cmd+="--micro-batch-size=$micro_batch_size "
cmd+="--num-gpus-per-node=$gpu_num_per_node "
cmd+="--num-nodes=1 "
cmd+="--node-ips=10.11.0.2 "

set -x

$cmd 
