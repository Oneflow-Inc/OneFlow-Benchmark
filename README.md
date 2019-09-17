# Oneflow-benchmark
OneFlow models for benchmarking.

## cnns
1 node, 1 gpu:
```
python cnn_benchmark/of_cnn_benchmarks.py \
--gpu_num_per_node=1 \
--model="vgg16" \
--batch_size=8 \
--iter_num=5 \
--learning_rate=0.01 \
--optimizer="sgd" \
--log_every_n_iter=1 \
--data_dir="/dataset/PNGS/PNG228/of_record_repeated"
```

2 nodes, 2 gpu each node:
```
python cnn_benchmark/of_cnn_benchmarks.py \
--gpu_num_per_node=2 \
--multinode \
--node_list="192.168.1.12,192.168.1.14" \
--model="vgg16" \
--batch_size=8 \
--iter_num=5 \
--learning_rate=0.01 \
--optimizer="sgd" \
--log_every_n_iter=1 \
--data_dir="/dataset/PNGS/PNG228/of_record_repeated"
```
