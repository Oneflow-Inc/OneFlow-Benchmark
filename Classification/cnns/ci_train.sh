NUM_EPOCH=${1:-50}
GPU_NUM=${2:-8}
NODE_NUM=${3:-1}
BATCH_SIZE=${4:-32}
LEARNING_RATE=${5:-1.536}
SRC_ROOT=${6:-"Classification/cnns"}
DATA_ROOT=${7:-"/dataset/ImageNet/ofrecord"}

test_case=n${NODE_NUM}_g${GPU_NUM}_b${BATCH_SIZE}_lr${LEARNING_RATE}_e${NUM_EPOCH}
LOG_FOLDER=./log
mkdir -p $LOG_FOLDER

model="resnet50"
LOGFILE=$LOG_FOLDER/${model}_${test_case}.log

export PYTHONUNBUFFERED=1
export NCCL_LAUNCH_MODE=PARALLEL

python3 ${SRC_ROOT}/of_cnn_train_val.py \
     --train_data_dir=$DATA_ROOT/train \
     --train_data_part_num=256 \
     --val_data_dir=$DATA_ROOT/validation \
     --val_data_part_num=256 \
     --num_nodes=${NODE_NUM} \
     --gpu_num_per_node=${GPU_NUM} \
     --optimizer="sgd" \
     --momentum=0.875 \
     --label_smoothing=0.1 \
     --learning_rate=${LEARNING_RATE} \
     --loss_print_every_n_iter=100 \
     --batch_size_per_device=${BATCH_SIZE} \
     --val_batch_size_per_device=50 \
     --use_fp16 \
     --channel_last=True \
     --pad_output \
     --fuse_bn_relu=True \
     --fuse_bn_add_relu=True \
     --nccl_fusion_threshold_mb=16 \
     --nccl_fusion_max_ops=24 \
     --gpu_image_decoder=True \
     --num_epoch=$NUM_EPOCH \
     --num_examples=1024 \
     --model=${model} 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
