#! /bin/bash
# set -ex

export ONEFLOW_GPT_NUM_GPUS_PER_NODE=8

# If you place training data on somewhere else, set this env
# export ONEFLOW_GPT_DATASET=/data/gpt/gpt_sample_dataset_text_document
export ONEFLOW_GPT_SEQ_LENGTH=2048

export ONEFLOW_GPT_HIDDEN_SIZE=1536
export ONEFLOW_GPT_NUM_ATTENTION_HEADS=16
export ONEFLOW_GPT_NUM_LAYERS=16

export ONEFLOW_GPT_TENSOR_MODEL_PARALLEL_SIZE=4
export ONEFLOW_GPT_PIPELINE_MODEL_PARALLEL_SIZE=1

export ONEFLOW_GPT_MICRO_BATCH_SIZE=8
export ONEFLOW_GPT_GLOBAL_BATCH_SIZE=16

source $(dirname $0)/pretrain.sh
