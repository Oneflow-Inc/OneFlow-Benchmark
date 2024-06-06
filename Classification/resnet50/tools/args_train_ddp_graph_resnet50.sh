rm -rf core.*

set -ex


# bash examples/args_train_ddp_graph.sh ${NUM_NODES} ${DEVICE_NUM_PER_NODE} ${NODE_RANK} ${MASTER_ADDR}
# ${OFRECORD_PATH} ${TRAIN_BATCH_SIZE} ${EPOCH} ${USE_FP16} ${PYTHON_BIN} ${RUN_TYPE} ${DEBUG_AND_NCCL} ${NSYS_BIN} ${RUN_COMMIT}

# bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 /dataset/79846248 192 50 false python3 ddp false '' 1

NUM_NODES=${1:-1}
DEVICE_NUM_PER_NODE=${2:-8}
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
OFRECORD_PATH=${5:-"/dataset/imagenet/ofrecord"}
TRAIN_BATCH_SIZE=${6:-192}
EPOCH=${7:-50}
USE_FP16=${8:-false}
PYTHON_BIN=${9:-"python3"}
RUN_TYPE=${10:-"ddp"} # graph+fp16
DECODE_TYPE=${11:-"cpu"}
PRINT_INTERVAL=${12:-100}
DEBUG_AND_NCCL=${13:-false}
NSYS_BIN=${14:-""}
RUN_COMMIT=${15:-"master"}
ACC=${16:-1}
VAL_BATCH_SIZE=${17:-50}


SRC_DIR=$(realpath $(dirname $0)/..)

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

TRAN_MODEL="resnet50"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$HOSTNAME/${NUM_NODES}n${DEVICE_NUM_PER_NODE}g
mkdir -p $LOG_FOLDER
LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_${RUN_TYPE}_DC${DECODE_TYPE}_${AMP_OR}_mb${TRAIN_BATCH_SIZE}_gb$((${TRAIN_BATCH_SIZE}*${NUM_NODES}*${DEVICE_NUM_PER_NODE}*${ACC}))_acc${ACC}_${NUM_NODES}n${DEVICE_NUM_PER_NODE}g_${RUN_COMMIT}_${RUN_TIME}


export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
#export ONEFLOW_COMM_NET_IB_ENABLE=True
export NCCL_LAUNCH_MODE=GROUP
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
echo DEBUG_AND_NCCL=$DEBUG_AND_NCCL
if $DEBUG_AND_NCCL; then
    export ONEFLOW_DEBUG_MODE=1
    echo ONEFLOW_DEBUG_MODE=$ONEFLOW_DEBUG_MODE
    export NCCL_DEBUG=INFO
    echo NCCL_DEBUG=$NCCL_DEBUG
fi

#export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
#export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1
#export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1
#export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1
#export ONEFLOW_STREAM_REUSE_CUDA_EVENT=1

#export ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC=true
#export ONEFLOW_VM_WORKLOAD_ON_SCHEDULER_THREAD=1

LEARNING_RATE=$(echo | awk "{print $NUM_NODES*$DEVICE_NUM_PER_NODE*$TRAIN_BATCH_SIZE*$ACC/1000}")
MOM=0.875
OFRECORD_PART_NUM=256

EXIT_NUM=-1

if [ ${EPOCH} -lt 10 ];then
    EXIT_NUM=300
fi
CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
    export ONEFLOW_DEBUG_MODE=1
    # CMD+="${NSYS_BIN} profile --stats true -t nvtx --output ${LOG_FILENAME} "
    export CUDNN_LOGINFO_DBG=1
    export CUDNN_LOGDEST_DBG=${SRC_DIR}/cudnn.log
    CMD+="${NSYS_BIN} profile --stats true --output ${LOG_FILENAME} "
    EXIT_NUM=30
fi


CMD+="${PYTHON_BIN} -m oneflow.distributed.launch "

CMD+="--nproc_per_node ${DEVICE_NUM_PER_NODE} "
CMD+="--nnodes ${NUM_NODES} "
CMD+="--node_rank ${NODE_RANK} "
CMD+="--master_addr ${MASTER_ADDR} "
CMD+="--master_port 12345 "
CMD+="${SRC_DIR}/train.py "
CMD+="--ofrecord-path ${OFRECORD_PATH} "
CMD+="--ofrecord-part-num ${OFRECORD_PART_NUM} "
CMD+="--num-devices-per-node ${DEVICE_NUM_PER_NODE} "
CMD+="--lr ${LEARNING_RATE} "
CMD+="--momentum ${MOM} "
CMD+="--num-epochs ${EPOCH} "
CMD+="--train-batch-size ${TRAIN_BATCH_SIZE} "
CMD+="--train-global-batch-size $((${TRAIN_BATCH_SIZE}*${NUM_NODES}*${DEVICE_NUM_PER_NODE}*${ACC})) "
CMD+="--val-batch-size ${VAL_BATCH_SIZE} "
CMD+="--val-global-batch-size $((${VAL_BATCH_SIZE}*${NUM_NODES}*${DEVICE_NUM_PER_NODE}*${ACC})) "
CMD+="--print-interval ${PRINT_INTERVAL} "
#CMD+="--synthetic-data "

if $USE_FP16; then
    echo USE_FP16=$USE_FP16
    CMD+="--use-fp16 --channel-last "
fi

if [ $EXIT_NUM != -1 ]; then
    CMD+="--skip-eval "
fi
if [ $RUN_TYPE == 'ddp' ]; then
    CMD+="--ddp "
else
    CMD+="--scale-grad --graph "
    CMD+="--fuse-bn-relu "
    CMD+="--fuse-bn-add-relu "
fi

if [ $DECODE_TYPE == 'gpu' ]; then
    CMD+="--use-gpu-decode "
fi

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

echo "Writting log to ${LOG_FILENAME}.log"

if [[ ! -z "${NSYS_BIN}" ]]; then
    rm ${LOG_FOLDER}/*.sqlite
    mkdir -p ${LOG_FILENAME}
    #rm -rf ./log/$HOSTNAME/oneflow.*
    cp ./log/$HOSTNAME/* ${LOG_FILENAME}/
    mv ${SRC_DIR}/cudnn.log ${LOG_FILENAME}/cudnn.log
fi

rm -rf ./log/$HOSTNAME
echo "done"
