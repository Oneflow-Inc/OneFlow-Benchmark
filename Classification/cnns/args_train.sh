rm -rf core.*
rm -rf ./output ./initial_model 

# bash args_train.sh ${NUM_NODES} ${NUM_GPUS_PER_NODE} ${BATCH_SIZE} ${USE_FP16} ${NUM_EPOCH} ${LOSS_PRINT_ITER} ${TRAIN_DATA_PATH} ${VAL_DATA_PATH} ${PYTHON_BIN} ${NODE_IPS} ${NSYS_BIN}

NUM_NODES=${1:-1}
NUM_GPUS_PER_NODE=${2:-8}
BATCH_SIZE=${3:-192}
USE_FP16=${4:-True}
NUM_EPOCH=${5:-2}
LOSS_PRINT_ITER=${6:-10}
TRAIN_DATA_PATH=${7:-""}
VAL_DATA_PATH=${8:-""}
PYTHON_BIN=${9:-"python3"}
NODE_IPS=${10:-"10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"}
DEBUG_AND_NCCL=${11:-False}
NSYS_BIN=${11:-""}


TRAN_MODEL="resnet50"
LOG_FOLDER=./output/logs/${NUM_NODES}n${NUM_GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_${NUM_NODES}n${NUM_GPUS_PER_NODE}g_b${BATCH_SIZE}_FP16${USE_FP16}_training.log

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
echo DEBUG_AND_NCCL=$DEBUG_AND_NCCL
if [ $DEBUG_AND_NCCL ]; then
    export ONEFLOW_DEBUG_MODE=1
    echo ONEFLOW_DEBUG_MODE=$ONEFLOW_DEBUG_MODE
    export NCCL_DEBUG=INFO
    echo NCCL_DEBUG=$NCCL_DEBUG
fi

CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    CMD+="${NSYS_BIN} profile --stats true --output ${TRAN_MODEL}_v0.4.0_${NUM_NODES}_${NUM_GPUS_PER_NODE}_%h_%p "
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

LEARNING_RATE=$(($NUM_GPUS_PER_NODE*$BATCH_SIZE/1000))

CMD+="--batch_size_per_device=${BATCH_SIZE} "
CMD+="--learning_rate=${LEARNING_RATE} "
CMD+="--loss_print_every_n_iter=${LOSS_PRINT_ITER} "

CMD+="--channel_last=False "
if [ $USE_FP16 ]; then
    echo USE_FP16=$USE_FP16
    CMD+="--use_fp16 --pad_output --channel_last=True "
fi

CMD+="--fuse_bn_relu=True "
CMD+="--fuse_bn_add_relu=True "
CMD+="--nccl_fusion_threshold_mb=16 "
CMD+="--nccl_fusion_max_ops=24 "
CMD+="--gpu_image_decoder=True "

CMD+="--num_epoch=${NUM_EPOCH} "
CMD+="--model=${TRAN_MODEL} "

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}

echo "Writting log to ${LOG_FILENAME}"
