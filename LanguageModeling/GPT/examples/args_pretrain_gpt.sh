#! /bin/bash
# set -ex

# Runs the "117M" parameter model

# bash args_pretrain_gpt.sh $NUM_NODES $NUM_GPUS_PER_NODE $M_P $P_P $MICRO_BATCH_SIZE $GLOABAL_BATCH_SIZE $USE_FP16 $TRAIN_ITERS $LOG_INTERVAL $DATA_PATH $NUM_LAYERS $HIDDEN_SIZE $NUM_ATTENTION_HEADS $SEQ_LENGTH $PYTHON_BIN $NODE_IPS $DEBUG_AND_NCCL $NSYS_BIN $RUN_COMMIT


NUM_NODES=${1:-1}
NUM_GPUS_PER_NODE=${2:-8}
M_P=${3:-1}
P_P=${4:-1}
MICRO_BATCH_SIZE=${5:-8}
GLOABAL_BATCH_SIZE=${6:-16}
USE_FP16=${7:-true}
TRAIN_ITERS=${8:-500}
LOG_INTERVAL=${9:-1}
DATA_PATH=${10:-""}
NUM_LAYERS=${11:-16}
HIDDEN_SIZE=${12:-1536}
NUM_ATTENTION_HEADS=${13:-16}
SEQ_LENGTH=${14:-2048}
PYTHON_BIN=${15:-"python3"}
NODE_IPS=${16:-"127.0.0.1"}
DEBUG_AND_NCCL=${17:-false}
NSYS_BIN=${18:-""}
RUN_COMMIT=${19:-1}

WORLD_SIZE=$(($NUM_GPUS_PER_NODE*$NUM_NODES))
D_P=$(($WORLD_SIZE/$M_P/$P_P))

RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=./output/logs/$HOSTNAME/${NUM_NODES}n${NUM_GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOG_FILENAME=$LOG_FOLDER/oneflow_gpt_${NUM_NODES}n${NUM_GPUS_PER_NODE}g_dp${D_P}_mp${M_P}_pp${P_P}_mbs${MICRO_BATCH_SIZE}_gbs${GLOABAL_BATCH_SIZE}_sql${SEQ_LENGTH}_l${NUM_LAYERS}_hsz${HIDDEN_SIZE}_ahs${NUM_ATTENTION_HEADS}_${RUN_COMMIT}_${RUN_TIME}.log

# save model
# CHECKPOINT_PATH=./model_save
# mkdir -p $CHECKPOINT_PATH


export PYTHONUNBUFFERED=1
# export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
# export ONEFLOW_DEBUG_MODE=1
# export NCCL_DEBUG=INFO

export NCCL_LAUNCH_MODE=GROUP

echo DEBUG_AND_NCCL=$DEBUG_AND_NCCL
if $DEBUG_AND_NCCL; then
    export ONEFLOW_DEBUG_MODE=1
    echo ONEFLOW_DEBUG_MODE=$ONEFLOW_DEBUG_MODE
    export NCCL_DEBUG=INFO
    echo NCCL_DEBUG=$NCCL_DEBUG
fi

if [[ ${NUM_NODES} -gt 1 ]]; then
    export ONEFLOW_COMM_NET_IB_ENABLE=1
fi

CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    CMD+="${NSYS_BIN} profile --stats true --output oneflow_gpt_${NUM_NODES}n${NUM_GPUS_PER_NODE}g_dp${D_P}_mp${M_P}_pp${P_P}_mbs${MICRO_BATCH_SIZE}_gbs${GLOABAL_BATCH_SIZE}_sql${SEQ_LENGTH}_l${NUM_LAYERS}_hsz${HIDDEN_SIZE}_ahs${NUM_ATTENTION_HEADS}_${RUN_COMMIT}_%h_%p "
fi

CMD+="${PYTHON_BIN} oneflow_gpt/training.py "

CMD+=" --num-layers ${NUM_LAYERS}"
CMD+=" --hidden-size ${HIDDEN_SIZE}"
CMD+=" --num-attention-heads ${NUM_ATTENTION_HEADS}"
CMD+=" --micro-batch-size ${MICRO_BATCH_SIZE}"

if [[ ! -z "${GLOABAL_BATCH_SIZE}" ]]; then
    CMD+=" --global-batch-size ${GLOABAL_BATCH_SIZE}"
fi

CMD+=" --tensor-model-parallel-size ${M_P}"
CMD+=" --pipeline-model-parallel-size ${P_P}"
CMD+=" --num-gpus-per-node ${NUM_GPUS_PER_NODE}"
CMD+=" --num-nodes ${NUM_NODES}"
CMD+=" --node-ips ${NODE_IPS}"
CMD+=" --train-iters ${TRAIN_ITERS}"
CMD+=" --dataset ${DATA_PATH}"
CMD+=" --seq-length ${SEQ_LENGTH}"
CMD+=" --vocab-size 50257"
CMD+=" --split 949,50,1"
CMD+=" --learning-rate 0.00015"
CMD+=" --min-lr 1.0e-5"
CMD+=" --lr-decay-style cosine"
CMD+=" --lr-decay-iters 320000"
CMD+=" --lr-warmup-fraction 0.01"
CMD+=" --optimizer adamw"
CMD+=" --weight-decay 1e-2"
CMD+=" --clip-grad 1.0"
# CMD+=" --save ${CHECKPOINT_PATH}"
CMD+=" --save-interval 100000"
CMD+=" --log-interval ${LOG_INTERVAL}"
CMD+=" --checkpoint-activations"
CMD+=" --multihead-attention-fusion"

if $USE_FP16; then
    echo USE_FP16=$USE_FP16
    CMD+=" --fp16"
fi


if [[ ! -z "${NSYS_BIN}" ]]; then
    CMD+=" --profile-transformer-layer"
fi


echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}

echo "Writting log to ${LOG_FILENAME}"
