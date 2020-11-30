log_dir=log
mkdir -p $log_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
gpu_num_per_node=8
n_head=16
n_embd=1024
test_case=gpt2_${gpu_num_per_node}_gpu
mem_file=$test_case.mem
log_file=$test_case.log
#python3 tools/gpu_memory_usage.py $gpu_num_per_node 1>$mem_file 2>&1 </dev/null &
#gdb --arg \

  python3 src/train.py \
    --num_nodes=1 \
    --node_ips="10.11.0.2,10.11.0.3" \
    --dataset=/datasets/wiki/enwiki/AA \
    --batch_size_per_device=2 \
    --gpu_num_per_node=$gpu_num_per_node \
    --seq_len=1024 \
    --optimizer=adam \
    --embedding_dropout=0.1 \
    --output_dropout=0.1 \
    --attention_dropout=0.1 \
    --n_vocab=50257 \
    --n_ctx=1024 \
    --n_embd=$n_embd \
    --n_head=$n_head \
    --n_layer=1 > $log_file