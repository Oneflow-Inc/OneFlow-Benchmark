#! /bin/bash

set -ex

export ONEFLOW_GPT_SEQ_LENGTH=1024
export ONEFLOW_GPT_NUM_LAYERS=12
export ONEFLOW_GPT_HIDDEN_SIZE=768
export ONEFLOW_GPT_NUM_ATTENTION_HEADS=12
export ONEFLOW_GPT_NUM_GPUS_PER_NODE=1

source `dirname $0`/pretrain_gpt.sh
