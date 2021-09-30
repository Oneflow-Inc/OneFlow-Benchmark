test_case=n${E2E_NODE_NUM}_g${E2E_GPU_NUM}_b${E2E_BATCH_SIZE}_lr${E2E_LEARNING_RATE}_e${E2E_NUM_EPOCH}
LOG_FOLDER=./log
mkdir -p $LOG_FOLDER

model="resnet50"
LOGFILE=$LOG_FOLDER/${model}_${test_case}.log

export PYTHONUNBUFFERED=1
export NCCL_LAUNCH_MODE=PARALLEL

python3 ${E2E_SRC_ROOT}/of_cnn_train_val.py \
     --train_data_dir=$E2E_DATA_ROOT/train \
     --train_data_part_num=256 \
     --val_data_dir=$E2E_DATA_ROOT/validation \
     --val_data_part_num=256 \
     --num_nodes=${E2E_NODE_NUM} \
     --gpu_num_per_node=${E2E_GPU_NUM} \
     --optimizer="sgd" \
     --momentum=0.875 \
     --label_smoothing=0.1 \
     --learning_rate=${E2E_LEARNING_RATE} \
     --loss_print_every_n_iter=100 \
     --batch_size_per_device=${E2E_BATCH_SIZE} \
     --val_batch_size_per_device=50 \
     --use_fp16 \
     --channel_last=True \
     --pad_output \
     --fuse_bn_relu=True \
     --fuse_bn_add_relu=True \
     --nccl_fusion_threshold_mb=16 \
     --nccl_fusion_max_ops=24 \
     --gpu_image_decoder=True \
     --num_epoch=$E2E_NUM_EPOCH \
     --num_examples=1024 \
     --model=${model} 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
