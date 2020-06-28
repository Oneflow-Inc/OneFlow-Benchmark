export ENABLE_USER_OP=True
rm -rf core.* 
rm -rf ./output/snapshots/*

DATA_ROOT=/dataset/ImageNet/ofrecord

python3 of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=256 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --gpu_num_per_node=4 \
    --model_update="momentum" \
    --learning_rate=0.256 \
    --loss_print_every_n_iter=20 \
    --batch_size_per_device=64 \
    --val_batch_size_per_device=125 \
    --num_epoch=90 \
    --model="resnet50" 