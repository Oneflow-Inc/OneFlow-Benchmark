rm -rf core.*
rm -rf ./output/snapshots/*

if [ -n "$1" ]; then
    NUM_EPOCH=$1
else
    NUM_EPOCH=50
fi
echo NUM_EPOCH=$NUM_EPOCH

# training with imagenet
if [ -n "$2" ]; then
    DATA_ROOT=$2
else
    DATA_ROOT=/data/imagenet/ofrecord
fi
echo DATA_ROOT=$DATA_ROOT

BATCH_SIZE=${3:-""}
echo BATCH_SIZE=$BATCH_SIZE

SSP_PLACEMENT=${4:-""}
echo SSP_PLACEMENT=$SSP_PLACEMENT

MODEL_NAME=${5:-"alexnet"}
echo MODEL_NAME=$MODEL_NAME

LOG_FOLDER=../logs
mkdir -p $LOG_FOLDER
LOGFILE=$LOG_FOLDER/resnet_training.log

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE

python3 of_ssp_cnn_train_val.py \
     --train_data_dir=$DATA_ROOT/train \
     --train_data_part_num=256 \
     --num_nodes=2 \
     --gpu_num_per_node=8 \
     --ssp_placement=$SSP_PLACEMENT \
     --optimizer="sgd" \
     --momentum=0.875 \
     --label_smoothing=0.1 \
     --learning_rate=0.768 \
     --loss_print_every_n_iter=100 \
     --batch_size_per_device=$BATCH_SIZE \
     --val_batch_size_per_device=50 \
     --channel_last=False \
     --fuse_bn_relu=True \
     --fuse_bn_add_relu=True \
     --nccl_fusion_threshold_mb=16 \
     --nccl_fusion_max_ops=24 \
     --gpu_image_decoder=True \
     --num_epoch=$NUM_EPOCH \
     --model=$MODEL_NAME 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
