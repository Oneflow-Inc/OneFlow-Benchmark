export ENABLE_USER_OP=True
rm -rf core.* 
rm -rf ./output/snapshots/*
DATA_ROOT=/DATA/disk1/ImageNet/ofrecord

  python3 cnn_e2e/of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=256 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --node_ips='11.11.1.12,11.11.1.14' \
    --gpu_num_per_node=8 \
    --model_update="momentum" \
    --learning_rate=0.05 \
    --loss_print_every_n_iter=200 \
    --batch_size_per_device=32 \
    --val_batch_size_per_device=125 \
    --use_new_dataloader=true \
    --num_epoch=150 \
    --model="mobilenet_v2" \
    --wd=0.00004 --warmup_epochs=0 --lr_decay='cosine' \
    --mom=0.9 \
    --use_boxing_v2=true 
