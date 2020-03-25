rm -rf core.*
#gdb --args \
#DATA_ROOT=/dataset/imagenet_10pics
DATA_ROOT=/dataset/imagenet_1pic
python cnn_e2e/test_data_loader.py \
  --data_train=$DATA_ROOT/mxnet/train.rec \
  --data_train_idx=$DATA_ROOT/mxnet/train.idx \
  --data_val=$DATA_ROOT/mxnet/train.rec \
  --data_val_idx=$DATA_ROOT/mxnet/train.idx \
  --train_data_dir=$DATA_ROOT/ofrecord \
  --train_data_part_num=1 \
  --val_data_dir=$DATA_ROOT/ofrecord \
  --val_data_part_num=1 \
  --val_batch_size_per_device=1 \
  --batch_size_per_device=1 \
  --num_examples=1 \
  --num_val_examples=1 \
  --gpu_num_per_node=1 \
  --num_epochs=1
