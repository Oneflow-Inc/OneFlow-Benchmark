BENCH_ROOT=$1
NUM_NODES=$2
GPU_NUM_PER_NODE=$3
BSZ_PER_DEVICE=$4

if [ -n "$5" ]; then
    DATA_ROOT=$5
else
    DATA_ROOT=/datasets/ImageNet/OneFlow
fi

DATA_PART_NUM=44

rm -rf ./log
mkdir ./log

NUM_ITERS=300
NUM_EXAMPLES=$(($NUM_NODES * $GPU_NUM_PER_NODE * $BSZ_PER_DEVICE * $NUM_ITERS))

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE

python3 $BENCH_ROOT/of_cnn_train_val.py \
    --num_examples=$NUM_EXAMPLES \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=$DATA_PART_NUM \
    --num_nodes=$NUM_NODES \
    --gpu_num_per_node=$GPU_NUM_PER_NODE \
    --optimizer="sgd" \
    --momentum=0.875 \
    --label_smoothing=0.1 \
    --learning_rate=0.001 \
    --loss_print_every_n_iter=100 \
    --batch_size_per_device=$BSZ_PER_DEVICE \
    --val_batch_size_per_device=125 \
    --num_epoch=1 \
    --log_dir=./log \
    --use_fp16 \
    --channel_last=True \
    --pad_output \
    --fuse_bn_relu=True \
    --fuse_bn_add_relu=True \
    --nccl_fusion_threshold_mb=16 \
    --nccl_fusion_max_ops=24 \
    --gpu_image_decoder=True \
    --model="resnet50"
