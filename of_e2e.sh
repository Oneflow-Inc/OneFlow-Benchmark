export ENABLE_USER_OP=True
rm -rf core.* 
rm -rf ./output/snapshots/*
#DATA_ROOT=/DATA/disk1/of_imagenet_example
#DATA_ROOT=/DATA/disk1/ImageNet/ofrecord
DATA_ROOT=/dataset/ImageNet/ofrecord
#DATA_ROOT=/dataset/imagenet-mxnet
  #python3 cnn_benchmark/of_cnn_train_val.py \
#nvprof -f -o resnet.nvvp \
#gdb --args \
  #python3 cnn_e2e/optimizer_util.py \
  python3 cnn_e2e/of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=256 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=1 \
    --node_ips='11.11.1.12,11.11.1.14' \
    --gpu_num_per_node=4 \
    --model_update="momentum" \
    --learning_rate=0.256 \
    --loss_print_every_n_iter=20 \
    --batch_size_per_device=32 \
    --val_batch_size_per_device=125 \
    --use_new_dataloader=true \
    --num_epoch=90 \
    --model="resnet50" 
    #--model_load_dir=/DATA/disk1/mx_imagenet/of_mx_mixed_init_model \
    #--model_load_dir=/DATA/disk1/baseline_test_logs/num_epochs_50/snapshots/snapshot_epoch_49 \
    #--wd=0.0 --warmup_epochs=0 --lr_decay=None \
    #--mom=0.875 \
    #--model_load_dir=/home/xiexuan/model_convert_resnet50/of_init_model \
    #--use_fp16 true \
    #--use_boxing_v2=true \
    # --train_data_dir=$DATA_ROOT/train \
    # --train_data_part_num=256 \
    # --val_data_dir=$DATA_ROOT/validation \
    # --val_data_part_num=256 \
    #--weight_l2=3.0517578125e-05 \
    #--num_examples=1024 \
    #--data_dir="/mnt/13_nfs/xuan/ImageNet/ofrecord/train"
    #--data_dir="/mnt/dataset/xuan/ImageNet/ofrecord/train"
    #--warmup_iter_num=10000 \
