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
    DATA_ROOT=/dataset/ImageNet/ofrecord
fi
echo DATA_ROOT=$DATA_ROOT

LOG_FOLDER=../logs
mkdir -p $LOG_FOLDER
LOGFILE=$LOG_FOLDER/resnet_training.log

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE

     #--momentum=0.875 \
python3 of_cnn_train_val.py \
     --train_data_dir=$DATA_ROOT/train \
     --train_data_part_num=256 \
     --val_data_dir=$DATA_ROOT/validation \
     --val_data_part_num=256 \
     --num_nodes=1 \
     --model_load_dir=/ssd/xiexuan/models/resnet50/init_ckpt \
     --gpu_num_per_node=1 \
     --optimizer="sgd" \
     --momentum=0.0 \
     --lr_decay="none" \
     --label_smoothing=0.1 \
     --learning_rate=0.1 \
     --loss_print_every_n_iter=1 \
     --batch_size_per_device=64 \
     --val_batch_size_per_device=64 \
     --channel_last=False \
     --pad_output \
     --fuse_bn_relu=True \
     --fuse_bn_add_relu=True \
     --nccl_fusion_threshold_mb=16 \
     --nccl_fusion_max_ops=24 \
     --gpu_image_decoder=True \
     --num_epoch=$NUM_EPOCH \
     --model="resnet50" 2>&1 | tee ${LOGFILE}
     # --use_fp16 \

echo "Writting log to ${LOGFILE}"
