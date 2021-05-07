#! /bin/bash

export ONEFLOW_GPT_SEQ_LENGTH=2048
export ONEFLOW_GPT_NUM_LAYERS=16
export ONEFLOW_GPT_HIDDEN_SIZE=1536
export ONEFLOW_GPT_NUM_ATTENTION_HEADS=16
export ONEFLOW_GPT_TENSOR_MODEL_PARALLEL_SIZE=8
export ONEFLOW_GPT_MICRO_BATCH_SIZE=8
export ONEFLOW_GPT_GLOBAL_BATCH_SIZE=32
export ONEFLOW_GPT_NUM_GPUS_PER_NODE=8
export ONEFLOW_GPT_NUM_NODES=2
export ONEFLOW_GPT_NODE_IPS="192.168.1.16,192.168.1.15"

source `dirname $0`/pretrain_gpt.sh
