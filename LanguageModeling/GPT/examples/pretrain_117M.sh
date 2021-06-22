#! /bin/bash
# set -ex

export ONEFLOW_GPT_NUM_GPUS_PER_NODE=1

# If you place training data on somewhere else, set this env
# export ONEFLOW_GPT_DATASET=/data/gpt/gpt_sample_dataset_text_document
export ONEFLOW_GPT_SEQ_LENGTH=1024

export ONEFLOW_GPT_NUM_LAYERS=12
export ONEFLOW_GPT_HIDDEN_SIZE=768
export ONEFLOW_GPT_NUM_ATTENTION_HEADS=12

export ONEFLOW_GPT_MICRO_BATCH_SIZE=8
export ONEFLOW_GPT_GLOBAL_BATCH_SIZE=8

source $(dirname $0)/pretrain.sh
