#!/bin/bash
rm -rf core.* 

# Set up  dataset root dir
DATA_ROOT=/dataset/ImageNet/ofrecord

# Set up model path, e.g. :  vgg16_of_best_model_val_top1_721  alexnet_of_best_model_val_top1_54762 
MODEL_LOAD_DIR="resnet_v15_of_best_model_val_top1_77318"

  python3  of_cnn_evaluate.py \
    --num_epochs=3 \
    --num_val_examples=50000 \
    --model_load_dir=$MODEL_LOAD_DIR  \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --node_ips='127.0.0.1' \
    --gpu_num_per_node=4 \
    --val_batch_size_per_device=64 \
    --model="resnet50"


# # # Validatation on imagenet
# DATA_ROOT=/datasets/ImageNet/ofrecord

# # Set up model path, e.g. : output/snapshots/model_save-20200831131226/snapshot_epoch_9
# MODEL_LOAD_DIR="saved_models/snapshots/model_save-20200905093837/snapshot_epoch_89"

#   python3  of_cnn_evaluate.py \
#     --num_epochs=3 \
#     --num_val_examples=50000 \
#     --model_load_dir=$MODEL_LOAD_DIR  \
#     --val_data_dir=$DATA_ROOT/validation \
#     --val_data_part_num=256 \
#     --num_nodes=1 \
#     --node_ips='127.0.0.1' \
#     --gpu_num_per_node=8 \
#     --val_batch_size_per_device=100 \
#     --model="resnet50"