export ENABLE_USER_OP=True
rm -rf core.* 
rm -rf ./output/snapshots/*
DATA_ROOT=/dataset/ImageNet/ofrecord
#Please change this to your data root.
python3 of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --val_data_dir=$DATA_ROOT/validation \
    --train_data_part_num=256 \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --gpu_num_per_node=1 \
    --model_update="rmsprop" \
    --epsilon=1 \
    --decay_rate=0.9 \
    --learning_rate=0.045 \
    --lr_decay="exponential" \
    --lr_decay_rate=0.94 \
    --loss_print_every_n_iter=10 \
    --batch_size_per_device=16 \
    --val_batch_size_per_device=16 \
    --num_epoch=100 \
    --use_fp16=false \
    --model="inceptionv3" \
    --image_size=299 \
    --resize_shorter=256 \
    --gradient_clipping=2 \