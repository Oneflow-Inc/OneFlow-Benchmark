export ENABLE_USER_OP=True
rm -rf core.* 
rm -rf ./output/snapshots/*

# DATA_ROOT=/dataset/ImageNet/ofrecord

python3 cnn_benchmark/of_cnn_train_val.py \
    --train_data_part_num=256 \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --gpu_num_per_node=1 \
    --model_update="momentum" \
    --mom=0.875 \
    --learning_rate=0.256 \
    --loss_print_every_n_iter=1 \
    --batch_size_per_device=8 \
    --val_batch_size_per_device=8 \
    --use_new_dataloader=false \
    --num_epoch=90 \
    --use_fp16=false \
    --use_boxing_v2=true \
    --model="resnet50" 

    # --node_ips='11.11.1.12,11.11.1.14' \
    # --model_load_dir=/DATA/disk1/mx_imagenet/of_mx_mixed_init_model \
    # --train_data_dir=$DATA_ROOT/train \
    # --train_data_part_num=256 \
    # --val_data_dir=$DATA_ROOT/validation \
    # --val_data_part_num=256 \
    #--num_examples=1024 \
