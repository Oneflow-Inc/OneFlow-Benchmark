#! /bin/bash
# set -ex

export ONEFLOW_GPT_NUM_GPUS_PER_NODE=8
export ONEFLOW_GPT_NUM_NODES=4
# Set this env for your training nodes ip
# export ONEFLOW_GPT_NODE_IPS="10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"

# If you place training data on somewhere else, set this env
# export ONEFLOW_GPT_DATASET=/data/gpt/gpt_sample_dataset_text_document
export ONEFLOW_GPT_SEQ_LENGTH=2048

export ONEFLOW_GPT_HIDDEN_SIZE=2304
export ONEFLOW_GPT_NUM_ATTENTION_HEADS=24
export ONEFLOW_GPT_NUM_LAYERS=24

export ONEFLOW_GPT_TENSOR_MODEL_PARALLEL_SIZE=8
export ONEFLOW_GPT_PIPELINE_MODEL_PARALLEL_SIZE=1

export ONEFLOW_GPT_MICRO_BATCH_SIZE=8
export ONEFLOW_GPT_GLOBAL_BATCH_SIZE=32

source $(dirname $0)/pretrain.sh
