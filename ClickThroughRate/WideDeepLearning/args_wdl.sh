#! /bin/bash
# set -ex

# bash args_wdl.sh 1 1 32 false 10000 100 /dataset/f9f659c5/train /dataset/f9f659c5/val python3 "" false "" master

NUM_NODES=${1:-1}
NUM_GPUS_PER_NODE=${2:-8}
BATCH_SIZE=${3:-32}
USE_FP16=${4:-true}
MAX_ITER=${5:-1100}
LOSS_PRINT_ITER=${6:-100}
TRAIN_DATA_PATH=${7:-""}
VAL_DATA_PATH=${8:-""}
PYTHON_BIN=${9:-"python3"}
NODE_IPS=${10:-"10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"}
DEBUG_AND_NCCL=${11:-false}
NSYS_BIN=${12:-""}
RUN_COMMIT=${13:-"master"}

# if [ $NUM_GPUS_PER_NODE -eq 1 ]; then
#   export CUDA_VISIBLE_DEVICES=$(($ITER_NUM-1))
# fi

SRC_DIR=$(realpath $(dirname $0))

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi


TRAN_MODEL="wdl"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$HOSTNAME/${NUM_NODES}n${NUM_GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_lazy_${AMP_OR}_b${BATCH_SIZE}_${NUM_NODES}n${NUM_GPUS_PER_NODE}g_${RUN_COMMIT}_${RUN_TIME}

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=GROUP
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
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


EXIT_NUM=-1

#if [ ${MAX_ITER} -lt 100000 ];then
#    EXIT_NUM=301
#fi

CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
    export ONEFLOW_DEBUG_MODE=1
    # CMD+="${NSYS_BIN} profile --stats true -t nvtx --output ${LOG_FILENAME} "
    CMD+="${NSYS_BIN} profile --stats true --output ${LOG_FILENAME} "
    MAX_ITER=30
    EXIT_NUM=31
fi
CMD+="${PYTHON_BIN} wdl_train_eval.py "

if [[ ! -z "${TRAIN_DATA_PATH}" ]]; then
    CMD+="--train_data_dir=${TRAIN_DATA_PATH} "
fi
CMD+="--train_data_part_num=256 "
CMD+="--train_part_name_suffix_length=5 "

if [[ ! -z "${VAL_DATA_PATH}" ]]; then
    CMD+="--eval_data_dir=${VAL_DATA_PATH} "
fi
CMD+="--eval_data_part_num=256 "
CMD+="--eval_part_name_suffix_length=5 "

CMD+="--num_nodes=${NUM_NODES} "
CMD+="--gpu_num=${NUM_GPUS_PER_NODE} "
CMD+="--node_ips=${NODE_IPS} "
CMD+="--train_part_name_suffix_length=5 "
CMD+="--hidden_units_num=2 "
CMD+="--dataset_format=ofrecord "
CMD+="--max_iter=${MAX_ITER} "
CMD+="--loss_print_every_n_iter=${LOSS_PRINT_ITER} "
CMD+="--eval_interval=10000 "
CMD+="--batch_size=${BATCH_SIZE} "
CMD+="--deep_embedding_vec_size=16 "
CMD+="--wide_vocab_size=2322444 "
CMD+="--deep_vocab_size=2322444 "


echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

echo "Writting log to ${LOG_FILENAME}.log"

if [[ ! -z "${NSYS_BIN}" ]]; then
    rm ${LOG_FOLDER}/*.sqlite
fi

rm -rf ./log/$HOSTNAME
echo "done"
