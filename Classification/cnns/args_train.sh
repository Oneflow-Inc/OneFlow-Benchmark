#! /bin/bash
# set -ex

# bash args_train.sh ${NUM_NODES} ${NUM_GPUS_PER_NODE} ${BATCH_SIZE} ${USE_FP16} ${NUM_EPOCH} ${LOSS_PRINT_ITER} ${TRAIN_DATA_PATH} ${VAL_DATA_PATH} ${PYTHON_BIN} ${NODE_IPS} ${DEBUG_AND_NCCL} ${NSYS_BIN} ${RUN_COMMIT}

NUM_NODES=${1:-1}
NUM_GPUS_PER_NODE=${2:-8}
BATCH_SIZE=${3:-192}
USE_FP16=${4:-true}
NUM_EPOCH=${5:-2}
LOSS_PRINT_ITER=${6:-100}
TRAIN_DATA_PATH=${7:-""}
VAL_DATA_PATH=${8:-""}
PYTHON_BIN=${9:-"python3"}
NODE_IPS=${10:-"10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"}
DEBUG_AND_NCCL=${11:-false}
NSYS_BIN=${12:-""}
RUN_COMMIT=${13:-"master"}
USE_GPUDECODE=${14:-true}

# if [ $NUM_GPUS_PER_NODE -eq 1 ]; then
#   export CUDA_VISIBLE_DEVICES=$(($ITER_NUM-1))
# fi

SRC_DIR=$(realpath $(dirname $0))

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi


TRAN_MODEL="resnet50"
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
#export ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC=true

if [[ ${NUM_NODES} -gt 1 ]]; then
    export ONEFLOW_COMM_NET_IB_ENABLE=1
fi


EXIT_NUM=-1

if [ ${NUM_EPOCH} -lt 3 ];then
    EXIT_NUM=301
fi

CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
    export ONEFLOW_DEBUG_MODE=1
    # CMD+="${NSYS_BIN} profile --stats true -t nvtx --output ${LOG_FILENAME} "
    CMD+="${NSYS_BIN} profile --stats true --output ${LOG_FILENAME} "
    EXIT_NUM=31
fi
CMD+="${PYTHON_BIN} of_cnn_train_val.py "

if [[ ! -z "${TRAIN_DATA_PATH}" ]]; then
    CMD+="--train_data_dir=${TRAIN_DATA_PATH} "
fi
CMD+="--train_data_part_num=256 "

if [[ ! -z "${VAL_DATA_PATH}" ]]; then
    CMD+="--val_data_dir=${VAL_DATA_PATH} "
fi

CMD+="--val_data_part_num=256 "
CMD+="--num_nodes=${NUM_NODES} "
CMD+="--gpu_num_per_node=${NUM_GPUS_PER_NODE} "
CMD+="--node_ips=${NODE_IPS} "

LEARNING_RATE=$(echo | awk "{print $NUM_NODES*$NUM_GPUS_PER_NODE*$BATCH_SIZE/1000}")

CMD+="--batch_size_per_device=${BATCH_SIZE} "
CMD+="--learning_rate=${LEARNING_RATE} "
CMD+="--loss_print_every_n_iter=${LOSS_PRINT_ITER} "

CMD+="--channel_last=False "
if $USE_FP16; then
    echo USE_FP16=$USE_FP16
    CMD+="--use_fp16 --pad_output --channel_last=True "
fi

CMD+="--fuse_bn_relu=True "
CMD+="--fuse_bn_add_relu=True "
CMD+="--nccl_fusion_threshold_mb=16 "
CMD+="--nccl_fusion_max_ops=24 "
CMD+="--gpu_image_decoder=False "

if $USE_GPUDECODE; then
    CMD+="--gpu_image_decoder=True "
fi

CMD+="--num_epoch=${NUM_EPOCH} "
CMD+="--model=${TRAN_MODEL} "
CMD+="--exit_iter=${EXIT_NUM} "

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

echo "Writting log to ${LOG_FILENAME}.log"

if [[ ! -z "${NSYS_BIN}" ]]; then
    rm ${LOG_FOLDER}/*.sqlite
    mkdir -p ${LOG_FILENAME}
    rm -rf ./output/$HOSTNAME/oneflow.*
    cp -r ./output/$HOSTNAME/* ${LOG_FILENAME}/
fi

rm -rf ./log/$HOSTNAME
rm -rf ./output/$HOSTNAME
echo "done"
