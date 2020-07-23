rm -rf core.* 
rm -rf ./output/snapshots/*

DATA_ROOT=data/imagenet/ofrecord

python3 of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --num_examples=50 \
    --train_data_part_num=1 \
    --val_data_dir=$DATA_ROOT/validation \
    --num_val_examples=50 \
    --val_data_part_num=1 \
    --num_nodes=1 \
    --gpu_num_per_node=1 \
    --model_update="momentum" \
    --learning_rate=0.001 \
    --loss_print_every_n_iter=1 \
    --batch_size_per_device=16 \
    --val_batch_size_per_device=16 \
    --num_epoch=10 \
    --model="resnet50"