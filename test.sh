rm -rf core.*
#gdb --args \
#DATA_ROOT=/mnt/13_nfs/xuan/ImageNet
DATA_ROOT=/dataset/imagenet-mxnet
python cnn_benchmark/dali.py \
  --data_train=$DATA_ROOT/mxnet/train.rec \
  --data_train_idx=$DATA_ROOT/mxnet/train.idx \
  --data_val=$DATA_ROOT/mxnet/val.rec \
  --data_val_idx=$DATA_ROOT/mxnet/val.idx \
  --val_batch_size_per_device=20 \
  --gpu_num_per_node=4 \
  --num_examples=1024 \
  --num_epochs=90
