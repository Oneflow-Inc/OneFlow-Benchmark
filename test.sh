rm -rf core.*
#gdb --args \
#DATA_ROOT=/mnt/13_nfs/xuan/ImageNet
DATA_ROOT=/mnt/13_nfs/xuan/ImageNet/mxnet
#DATA_ROOT=/dataset/imagenet-mxnet
python cnn_benchmark/dali.py \
  --data_train=$DATA_ROOT/train.rec \
  --data_train_idx=$DATA_ROOT/train.idx \
  --data_val=$DATA_ROOT/val.rec \
  --data_val_idx=$DATA_ROOT/val.idx \
  --val_batch_size_per_device=20 \
  --loss_print_every_n_iter=200 \
  --gpu_num_per_node=4 \
  --num_epochs=90
  #--num_examples=1024 \
