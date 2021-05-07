#! /bin/bash

dataset=${ONEFLOW_GPT_DATASET:-"/data/gpt/gpt_sample_dataset_text_document"}
seq_length=${ONEFLOW_GPT_SEQ_LENGTH:-"2048"}

num_layers=${ONEFLOW_GPT_NUM_LAYERS:-"16"}
hidden_size=${ONEFLOW_GPT_HIDDEN_SIZE:-"1536"}
num_attn_heads=${ONEFLOW_GPT_NUM_ATTENTION_HEADS:-"16"}

micro_batch_size=${ONEFLOW_GPT_MICRO_BATCH_SIZE:-"8"}
global_batch_size=${ONEFLOW_GPT_GLOBAL_BATCH_SIZE}
tensor_model_parallel_size=${ONEFLOW_GPT_TENSOR_MODEL_PARALLEL_SIZE}
pipeline_model_parallel_size=${ONEFLOW_GPT_PIPELINE_MODEL_PARALLEL_SIZE}
num_accumulation_steps=${ONEFLOW_GPT_NUM_ACCUMULATION_STEPS}

num_gpus_per_node=${ONEFLOW_GPT_NUM_GPUS_PER_NODE:-"4"}
num_nodes=${ONEFLOW_GPT_NUM_NODES:-"1"}
node_ips=${ONEFLOW_GPT_NODE_IPS:-"10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"}

train_iters=${ONEFLOW_GPT_TRAIN_ITERS:-"500000"}

cmd=""

if [[ ! -z "${ONEFLOW_GTP_PROFILE_FILE}" ]]; then
    cmd+="nsys profile --stats true --output ${ONEFLOW_GTP_PROFILE_FILE} "
fi

if [[ ! -z "${ONEFLOW_GTP_GDB}" ]]; then
    cmd+="gdb --args "
fi

cmd+="python3 -m oneflow_gpt.training"
cmd+=" --num-layers ${num_layers}"
cmd+=" --hidden-size ${hidden_size}"
cmd+=" --num-attention-heads ${num_attn_heads}"
cmd+=" --micro-batch-size ${micro_batch_size}"

if [[ ! -z "${global_batch_size}" ]]; then
    cmd+=" --global-batch-size ${global_batch_size}"
fi

if [[ ! -z "${tensor_model_parallel_size}" ]]; then
    cmd+=" --tensor-model-parallel-size ${tensor_model_parallel_size}"
fi

if [[ ! -z "${pipeline_model_parallel_size}" ]]; then
    cmd+=" --pipeline-model-parallel-size ${pipeline_model_parallel_size}"
fi

if [[ ! -z "${num_accumulation_steps}" ]]; then
    cmd+=" --num-accumulation-steps ${num_accumulation_steps}"
fi

cmd+=" --num-gpus-per-node ${num_gpus_per_node}"
cmd+=" --num-nodes ${num_nodes}"
cmd+=" --node-ips ${node_ips}"
cmd+=" --train-iters ${train_iters}"
cmd+=" --learning-rate 0.00015"
cmd+=" --min-lr 1.0e-5"
cmd+=" --lr-decay-style cosine"
cmd+=" --lr-decay-iters 320000"
cmd+=" --lr-warmup-fraction 0.01"
cmd+=" --optimizer adamw"
cmd+=" --weight-decay 1e-2"
cmd+=" --clip-grad 1.0"
cmd+=" --dataset ${dataset}"
cmd+=" --seq-length ${seq_length}"
cmd+=" --vocab-size 50257"
cmd+=" --split 949,50,1"
cmd+=" --save model_save"
cmd+=" --save-interval 10000"
cmd+=" --log-interval 100"
cmd+=" --metric-print-format table"
cmd+=" --checkpoint-activations"
cmd+=" --multihead-attention-fusion"
cmd+=" --fp16"

if [[ ! -z "${ONEFLOW_GTP_PROFILE_FILE}" ]]; then
    cmd+=" --profile-transformer-layer"
fi

${cmd}
