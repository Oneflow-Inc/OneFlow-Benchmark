
OFRECORD_PATH="ofrecord"
if [ ! -d "$OFRECORD_PATH" ]; then
    wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord.tar.gz
    tar zxf imagenette_ofrecord.tar.gz
fi

MODEL_LOAD_DIR="initial_model_remove_mom"
CLASSES=10

python3 of_cnn_train_val.py \
    --train_data_dir=$OFRECORD_PATH/train \
    --val_data_dir=$OFRECORD_PATH/val \
    --train_data_part_num=1 \
    --val_data_part_num=1 \
    --num_nodes=1 \
    --gpu_num_per_node=1 \
    --optimizer="sgd" \
    --momentum=0.9 \
    --learning_rate=0.01 \
    --pad_output=False \
    --loss_print_every_n_iter=1 \
    --batch_size_per_device=512 \
    --val_batch_size_per_device=512 \
    --num_examples=9469 \
    --num_val_examples=3925 \
    --num_epoch=90 \
    --use_fp16=false \
    --model="alexnet" \
    --num_classes=$CLASSES \
    --model_load_dir=$MODEL_LOAD_DIR
