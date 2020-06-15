# OneFlow-Benchmark
OneFlow models for benchmarking.

## CNNs
### Train
* 1 node, 1 gpu:
    ```
    python3 cnn_benchmark/of_cnn_train_val.py \
        --gpu_num_per_node=1 \
        --batch_size_per_device=32 \
        --val_batch_size_per_device=32 \
        --use_new_dataloader=True \
        --train_data_part_num=256 \
        --val_data_part_num=256 \
        --num_epochs=1 \
        --model_update="momentum" \
        --learning_rate=0.256 \
        --use_fp16=False \
        --use_boxing_v2=True \
        --model="resnet50" 
    ```

* 2 nodes:

    simply add `--num_nodes=2 --node_ips="192.168.1.12,192.168.1.14" ` :

    ```
       python3 cnn_benchmark/of_cnn_train_val.py \
        --num_nodes=2 \
        --node_ips="192.168.1.12,192.168.1.14" \
        --gpu_num_per_node=1 \
        --batch_size_per_device=32 \
        --val_batch_size_per_device=32 \
        --use_new_dataloader=True \
        --train_data_part_num=256 \
        --val_data_part_num=256 \
        --num_epochs=1 \
        --model_update="momentum" \
        --learning_rate=0.256 \
        --use_fp16=False \
        --use_boxing_v2=True \
        --model="resnet50" 
 
    ```
### Validation
```
python3 cnn_benchmark/of_cnn_val.py \
    --model_load_dir=output/snapshots_0323 \
    --val_data_dir=$DATA_ROOT/validation \
    --val_data_part_num=256 \
    --gpu_num_per_node=4 \
    --loss_print_every_n_iter=20 \
    --val_batch_size_per_device=125 \
    --model="resnet50"
```

## BERT
### Pretrain
* 1 node, 1gpu:
    
    * bert base:
    ```
    python3 bert_benchmark/run_pretraining.py \
    --gpu_num_per_node=1 \
    --learning_rate=1e-4 \
    --batch_size_per_device=12 \
    --iter_num=5 \
    --loss_print_every_n_iter=1 \
    --seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_hidden_layers=12 \
    --num_attention_heads=12 \
    --max_position_embeddings=512 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --hidden_size_per_head=64 \
    --data_part_num=1 \
    --data_dir="/dataset/bert/of_wiki_seq_len_128" 
    ```

    * bert large:
    bert large's is different from bert base in flowing configurations: 
    `--max_predictions_per_seq=80 --num_hidden_layers=24 --num_attention_heads=16 --max_position_embeddings=512`

    ```
    python3 bert_benchmark/run_pretraining.py \
    --gpu_num_per_node=1 \
    --learning_rate=1e-4 \
    --batch_size_per_device=12 \
    --iter_num=5 \
    --loss_print_every_n_iter=1 \
    --seq_length=128 \
    --max_predictions_per_seq=80 \
    --num_hidden_layers=24 \
    --num_attention_heads=16 \
    --max_position_embeddings=512 \
    --type_vocab_size=2 \
    --vocab_size=30522 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --hidden_size_per_head=64 \
    --data_part_num=1 \
    --data_dir="/dataset/bert/of_wiki_seq_len_128"
    ```

* 2 nodes:

    simply add `--num_nodes=2 --node_ips="192.168.1.12,192.168.1.14" ` :


## build docker images from wheel
please put oneflow `*.whl` in docker/wheel folder, then build docker image use:
```
sh docker/build.sh
```

run docker image use:
```
sh docker/launch.sh
```
