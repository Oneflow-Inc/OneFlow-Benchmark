rm -rf core.* 
DATA_ROOT=/dataset/ImageNet/ofrecord
#DATA_ROOT=/dataset/imagenet-mxnet
  #python3 cnn_benchmark/of_cnn_train_val.py \
#gdb --args \
#nvprof -f -o resnet.nvvp \
  python3 cnn_e2e/of_cnn_train_val.py \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=256 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --num_nodes=2 \
    --node_ips='11.11.1.13,11.11.1.14' \
    --gpu_num_per_node=4 \
    --optimizer="momentum-cosine-decay" \
    --learning_rate=0.256 \
    --loss_print_every_n_iter=20 \
    --batch_size_per_device=32 \
    --val_batch_size_per_device=125 \
    --model="resnet50" 
    #--use_fp16 true \
    #--weight_l2=3.0517578125e-05 \
    #--num_examples=1024 \
    #--optimizer="momentum-decay" \
    #--data_dir="/mnt/13_nfs/xuan/ImageNet/ofrecord/train"
    #--data_dir="/mnt/dataset/xuan/ImageNet/ofrecord/train"
    #--warmup_iter_num=10000 \
