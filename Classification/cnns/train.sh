#!/bin/bash
rm -rf core.* 
rm -rf ./output/snapshots/*


# training with synthetic data
python3 of_cnn_train_val.py \
    --num_examples=50 \
    --num_val_examples=50 \
    --num_nodes=1 \
    --gpu_num_per_node=1 \
    --optimizer="sgd" \
    --momentum=0.875 \
    --learning_rate=0.001 \
    --loss_print_every_n_iter=1 \
    --batch_size_per_device=16 \
    --val_batch_size_per_device=16 \
    --num_epoch=10 \
    --model="resnet50"

