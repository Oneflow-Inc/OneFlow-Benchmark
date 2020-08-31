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



# # Validatation on mini-imagenet
# DATA_ROOT=data/mini-imagenet/ofrecord

# # Set up model path, e.g. : output/snapshots/model_save-20200831131226/snapshot_epoch_9
# MODEL_LOAD_DIR="output/snapshots/model_save-20200831131226/snapshot_epoch_9"

#   python3  of_cnn_evaluate.py \
#     --num_epochs=3 \
#     --num_val_examples=50 \
#     --model_load_dir=$MODEL_LOAD_DIR  \
#     --val_data_dir=$DATA_ROOT/validation \
#     --val_data_part_num=1 \
#     --num_nodes=1 \
#     --node_ips='127.0.0.1' \
#     --gpu_num_per_node=1 \
#     --val_batch_size_per_device=32 \
#     --model="resnet50"
