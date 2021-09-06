#! /bin/bash
# set -ex

export ONEFLOW_GPT_NUM_GPUS_PER_NODE=2
export ONEFLOW_GPT_NUM_NODES=2
# Set this env for your training nodes ip
export ONEFLOW_GPT_NODE_IPS="10.10.120.201,10.10.120.202"

export NCCL_SOCKET_IFNAME=enp49s0f0
export NCCL_DEBUG=INFO 

# If you place training data on somewhere else, set this env
export ONEFLOW_GPT_DATASET=/public/data/gpt_sample_dataset_text_document
export ONEFLOW_GPT_SEQ_LENGTH=2048

export ONEFLOW_GPT_HIDDEN_SIZE=1536
export ONEFLOW_GPT_NUM_ATTENTION_HEADS=16
export ONEFLOW_GPT_NUM_LAYERS=16

export ONEFLOW_GPT_TENSOR_MODEL_PARALLEL_SIZE=4
export ONEFLOW_GPT_PIPELINE_MODEL_PARALLEL_SIZE=1

export ONEFLOW_GPT_MICRO_BATCH_SIZE=8
export ONEFLOW_GPT_GLOBAL_BATCH_SIZE=16

source $(dirname $0)/pretrain.sh
