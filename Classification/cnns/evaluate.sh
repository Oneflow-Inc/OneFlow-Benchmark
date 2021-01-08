rm -rf core.* 

# Set up  dataset root dir
DATA_ROOT=/dataset/ImageNet/ofrecord

# Set up model path, e.g. :  vgg16_of_best_model_val_top1_721  alexnet_of_best_model_val_top1_54762 
MODEL_LOAD_DIR="resnet_v15_of_best_model_val_top1_77318"

PYTHONPATH=/home/dev/files/repos/oneflow6/build-release/python_scripts  \
python3  of_cnn_evaluate.py \
    --num_epochs=1 \
    --num_val_examples=50000 \
    --model_load_dir=$MODEL_LOAD_DIR  \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --node_ips='127.0.0.1' \
    --gpu_num_per_node=1 \
    --val_batch_size_per_device=320 \
    --model="resnet50" \
    --use_tensorrt=True \
    --use_int8_online=True \
    --use_int8_offline=False \
    |& tee rn50-offline-int8.log
