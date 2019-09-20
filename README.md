# Oneflow-benchmark
OneFlow models for benchmarking.

## cnns
1 node, 1 gpu:
```
python cnn_benchmark/of_cnn_benchmarks.py \
--gpu_num_per_node=1 \
--model="vgg16" \
--batch_size_per_device=8 \
--iter_num=5 \
--learning_rate=0.01 \
--optimizer="sgd" \
--loss_print_every_n_iter=1 \
--data_dir="/dataset/PNGS/PNG228/of_record_repeated"
```

2 nodes, 2 gpu each node:
```
python cnn_benchmark/of_cnn_benchmarks.py \
--gpu_num_per_node=2 \
--node_num=2 \
--node_list="192.168.1.12,192.168.1.14" \
--model="vgg16" \
--batch_size_per_device=8 \
--iter_num=5 \
--learning_rate=0.01 \
--optimizer="sgd" \
--loss_print_every_n_iter=1 \
--data_dir="/dataset/PNGS/PNG228/of_record_repeated"
```

## bert pretrain
1 node, 1gpu:
```
python bert_benchmark/run_pretraining.py \
--gpu_num_per_node=1 \
--node_num=1 \
--learning_rate=1e-4 \
--weight_l2=0.01 \
--batch_size_per_device=24 \
--iter_num=5 \
--loss_print_every_n_iter=1 \
--data_dir="/dataset/bert/of_wiki_seq_len_128" \
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

2 nodes, 2 gpu each node:
```
python bert_benchmark/run_pretraining.py \
--gpu_num_per_node=2 \
--node_num=2 \
--node_list='192.168.1.12,192.168.1.14' \
--learning_rate=1e-4 \
--weight_l2=0.01 \
--batch_size_per_device=24 \
--iter_num=15 \
--loss_print_every_n_iter=1 \
--data_dir="/dataset/bert/of_wiki_seq_len_128" \
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

