rm -rf core.* 
rm -rf ./output/snapshots/*


# # nsys profile -o debug5 --export text --force-overwrite true \
# python3 of_cnn_train_val.py \
#     --num_examples=4096 \
#     --num_val_examples=256 \
#     --num_nodes=1 \
#     --gpu_num_per_node=4 \
#     --model_update="momentum" \
#     --learning_rate=0.001 \
#     --loss_print_every_n_iter=1 \
#     --batch_size_per_device=32 \
#     --num_epoch=5 \
#     --model="resnet50"

#Please change $DATA_ROOT this to your own data root.
# python3 of_cnn_train_val.py \
#     --train_data_part_num=256 \
#     --val_data_part_num=256 \
#     --num_nodes=1 \
#     --gpu_num_per_node=4 \
#     --model_update="momentum" \
#     --mom=0.9 \
#     --learning_rate=0.01 \
#     --loss_print_every_n_iter=1 \
#     --batch_size_per_device=512 \
#     --val_batch_size_per_device=512 \
#     --num_epoch=5 \
#     --use_fp16=false \
#     --model="alexnet"

#Please change $DATA_ROOT this to your own data root.
# nsys profile -o vgg98 --force-overwrite true \
python3 of_cnn_train_val.py \
    --train_data_part_num=64 \
    --val_data_part_num=64 \
    --num_nodes=1 \
    --gpu_num_per_node=4 \
    --model_update="momentum" \
    --mom=0.9 \
    --learning_rate=0.01 \
    --loss_print_every_n_iter=1 \
    --batch_size_per_device=16 \
    --val_batch_size_per_device=16 \
    --num_epoch=5 \
    --use_fp16=false \
    --model="vgg"

# #Please change $DATA_ROOT this to your own data root.
# python3 of_cnn_train_val.py \
#     --train_data_part_num=256 \
#     --val_data_part_num=256 \
#     --num_nodes=1 \
#     --gpu_num_per_node=4 \
#     --model_update="rmsprop" \
#     --epsilon=1 \
#     --decay_rate=0.9 \
#     --learning_rate=0.045 \
#     --lr_decay="exponential" \
#     --lr_decay_rate=0.94 \
#     --lr_decay_epochs=2 \
#     --loss_print_every_n_iter=1 \
#     --batch_size_per_device=256 \
#     --val_batch_size_per_device=256 \
#     --num_epoch=5 \
#     --use_fp16=false \
#     --model="inceptionv3" \
#     --image_size=299 \
#     --resize_shorter=299 \
#     --gradient_clipping=2 \
#     --warmup_epochs=0