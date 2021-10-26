DEVICE_NUM_PER_NODE=2
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

# export CUDA_VISIBLE_DEVICES=3
python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    main.py \
        --deep_vocab_size 1603616 \
        --wide_vocab_size 1603616 \
        --hidden_units_num 7 \
        --hidden_size 1024 \
        --deep_embedding_vec_size 16 \
        --batch_size 32 \
        --print_interval 1 \
        --deep_dropout_rate 0 \
        --max_iter 100 \
        --execution_mode 'eager' \
        --ddp \
        --test_name 'n1g8_ddp' \
        --model_load_dir "/home/shiyunxiao/checkpoint_new" \

        
