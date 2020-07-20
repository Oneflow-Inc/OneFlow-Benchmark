export ENABLE_USER_OP=True
rm -rf core.* 
rm -rf ./output/snapshots/*

DATA_ROOT=/dataset/ImageNet/ofrecord

python3 cnn_benchmark/of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --val_data_dir=$DATA_ROOT/validation \
    --train_data_part_num=256 \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --gpu_num_per_node=1 \
    --model_update="momentum" \
    --mom=0.9 \
    --learning_rate=0.01 \
    --loss_print_every_n_iter=100 \
    --batch_size_per_device=512 \
    --val_batch_size_per_device=512 \
    --num_epoch=90 \
    --use_fp16=false \
    --use_boxing_v2=false \
    --model="alexnet" \
