Table of Contents
=================

* [Branch Notes](#branch-notes)
* [OneFlow\-Benchmark](#oneflow-benchmark)
  * [Classification](#classification)
  * [BERT](#bert)
    * [Pretrain](#pretrain)
  * [Wide Deep Learning](#wide-deep-learning)
  * [Generative](#generative)

# Branch Notes

请大家管理一下自己的分支，有用的分支写在下面，说明用途。过时没用的分支及时删除。

* (default)of_develop_py3: fllow oneflow:devevop, python3
* (useful)of_develop_py3_4cnns_back: 
    1) vgg16,alexnet,inceptionv3 old backup, fllow develop, still useful.
    2) of_cnn_val.py backup.
    3) dockerfile backup.
* ...




# OneFlow-Benchmark
OneFlow models for benchmarking.

## Classification
see: https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/of_develop_py3/Classification/resnet50v1.5

## BERT
### Pretrain
* 1 node, 1gpu:
    
    * bert base:
    ```
    python3 LanguageModeling/BERT/run_pretraining.py \
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
    python3 LanguageModeling/BERT/run_pretraining.py \
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

## Wide Deep Learning

## Generative

项目由一流科技, 之江实验室研发.
