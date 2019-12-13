# OneFlow-Benchmark
OneFlow models for benchmarking.

## Training
### cnns
* 1 node, 1 gpu:
    ```
    python3 cnn_benchmark/of_cnn_benchmarks.py \
    --gpu_num_per_node=1 \
    --model="vgg16" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --learning_rate=0.01 \
    --optimizer="sgd" \
    --loss_print_every_n_iter=1 \
    --warmup_iter_num=2 \
    --data_dir="/dataset/ofrecord/imagenet/train"
    ```

* 2 nodes, 2 gpu each node:

    simply add `--node_num=2 --node_list="192.168.1.12,192.168.1.14" ` :

    ```
    python3 cnn_benchmark/of_cnn_benchmarks.py \
    --gpu_num_per_node=2 \
    --node_num=2 \
    --node_list="192.168.1.12,192.168.1.14" \
    --model="vgg16" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --learning_rate=0.01 \
    --optimizer="sgd" \
    --loss_print_every_n_iter=1 \
    --warmup_iter_num=2 \
    --data_dir="/dataset/ofrecord/imagenet/train"
    ```

### bert pretrain
* 1 node, 1gpu:
    * bert base:
    ```
    python3 bert_benchmark/run_pretraining.py \
    --gpu_num_per_node=1 \
    --node_num=1 \
    --learning_rate=1e-4 \
    --weight_l2=0.01 \
    --batch_size_per_device=24 \
    --iter_num=5 \
    --loss_print_every_n_iter=1 \
    --warmup_iter_num=2 \
    --data_dir="/dataset/ofrecord/wiki_128" \
    --data_part_num=1 \
    --seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_hidden_layers=12 \
    --num_attention_heads=12 \
    --max_position_embeddings=512 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --hidden_size_per_head=64
    ```

    * bert large:
    ```
    python3 ../bert_benchmark/run_pretraining.py \
    --gpu_num_per_node=$GPU_NUM_PER_NODE \
    --node_num=$NODE_NUM \
    --node_list=$NODE_LIST \
    --learning_rate=1e-4 \
    --weight_l2=0.01 \
    --batch_size_per_device=24 \
    --iter_num=5 \
    --loss_print_every_n_iter=1 \
    --warmup_iter_num=2 \
    --seq_length=512 \
    --max_predictions_per_seq=80 \
    --num_hidden_layers=24 \
    --num_attention_heads=16 \
    --max_position_embeddings=512 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --hidden_size_per_head=64 \
    --data_part_num=32 \
    --data_dir=/dataset/ofrecord/wiki_512"
    ```

* 2 nodes, 2 gpu each node:

    simply add `--node_num=2 --node_list='192.168.1.12,192.168.1.14' `,see above.

## Inference
* 1 node, 1 gpu:
    ```
    python3 cnn_benchmark/of_cnn_infer_benchmarks.py \
    --gpu_num_per_node=1 \
    --model="vgg16" \
    --batch_size_per_device=8 \
    --iter_num=5 \
    --print_every_n_iter=1 \
    --warmup_iter_num=2 \
    --data_dir="/dataset/ofrecord/imagenet/train"
    ```
    If you want to run the benchmark with TensorRT or XLA, only pass `--use_tensorrt` or `--use_xla_jit` to enable it. Low-precision arithmetics such as float16 or int8 are usually faster than 32bit float, and you can pass `--precision=float16` for acceleration.


## build docker images from wheel
please put oneflow `*.whl` in docker/wheel folder, then build docker image use:
```
sh docker/build.sh
```

run docker image use:
```
sh docker/launch.sh
```
