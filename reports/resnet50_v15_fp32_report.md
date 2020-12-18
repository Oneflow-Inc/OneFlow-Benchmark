# OneFlow ResNet50-V1.5 Benchmark Test Report
This document reports OneFlow ResNet50-V1.5 benchmark test results on Aug 8 2020. 

## Test Environment
All tests were performed on 4 GPU Servers with 8x Tesla V100-SXM2-16GB and following is the main hardware and software configurations for each:  
- Tesla V100-SXM2-16GB x 8
- InfiniBand 100 Gb/sec (4X EDR)， Mellanox Technologies MT27700 Family
- 48 CPU(s), Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz
- Memory 384G
- Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-116-generic x86_64)
- CUDA Version: 10.2, Driver Version: 440.33.01
- OneFlow: v0.1.8, fix_infer_out_logical_blob_desc@17a2bdc9b
- OneFlow-Benchmark: master@892f87e6
- `nvidia-smi topo -m`
```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  CPU Affinity
GPU0     X      NV1     NV1     NV2     NV2     SYS     SYS     SYS     NODE    0-11,24-35
GPU1    NV1      X      NV2     NV1     SYS     NV2     SYS     SYS     NODE    0-11,24-35
GPU2    NV1     NV2      X      NV2     SYS     SYS     NV1     SYS     PIX     0-11,24-35
GPU3    NV2     NV1     NV2      X      SYS     SYS     SYS     NV1     PIX     0-11,24-35
GPU4    NV2     SYS     SYS     SYS      X      NV1     NV1     NV2     SYS     12-23,36-47
GPU5    SYS     NV2     SYS     SYS     NV1      X      NV2     NV1     SYS     12-23,36-47
GPU6    SYS     SYS     NV1     SYS     NV1     NV2      X      NV2     SYS     12-23,36-47
GPU7    SYS     SYS     SYS     NV1     NV2     NV1     NV2      X      SYS     12-23,36-47
mlx5_0  NODE    NODE    PIX     PIX     SYS     SYS     SYS     SYS      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

```

## Test Descriptions
Two groups of tests were performed with different batch size per device: 128 and 160.

Each group includes 6 tests with different number of devices: 1, 2, 4, 8, 16, 32.

`Throughput` of images/sec and `GPU Memory Usage` were logged and recorded.

Data type of all tests is `Float32`, XLA is not applied.

## Test Scripts
Please clone or download `cnns` folder from [OneFlow-Benchmark repository](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns). 

We create two bash scripts alone side with `cnns` folder for this test:
1. `local_run.sh` - launch a local oneflow with specific number of nodes and gpu number per node
```bash
# local_run.sh
NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BENCH_ROOT_DIR=cnns

DATA_ROOT=/path/to/ofrecord
rm -rf ./log
mkdir ./log

BSZ_PER_DEVICE=128 
#BSZ_PER_DEVICE=160

NUM_ITERS=200
NUM_EXAMPLES=$(($NUM_NODES * $GPU_NUM_PER_NODE * $BSZ_PER_DEVICE * $NUM_ITERS))

python3 ./$BENCH_ROOT_DIR/of_cnn_train_val.py \
    --num_examples=$NUM_EXAMPLES \
    --train_data_dir=$DATA_ROOT/train \
    --train_data_part_num=44 \
    --num_nodes=$NUM_NODES \
    --gpu_num_per_node=$GPU_NUM_PER_NODE \
    --model_update="momentum" \
    --learning_rate=0.001 \
    --loss_print_every_n_iter=20 \
    --batch_size_per_device=$BSZ_PER_DEVICE \
    --val_batch_size_per_device=125 \
    --num_epoch=1 \
    --log_dir=./log \
    --node_ips='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5' \
    --model="resnet50"
```
2. `launch_all.sh` - launch oneflow on all remote nodes with specific number of nodes and gpu number per node.
```bash
# launch_all.sh
#!/bin/bash

NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BENCH_ROOT_DIR=cnn3
LOCAL_RUN=local_run.sh

##############################################
#0 prepare the host list for training
#comment unused hosts with `#`
#or use first arg to limit the hosts number

declare -a host_list=("10.11.0.2" "10.11.0.3" "10.11.0.4" "10.11.0.5")

if [ -n "$1" ]
then
  host_num=$1
else
  host_num=${#host_list[@]}
fi

if [ ${host_num} -gt ${#host_list[@]} ]
then
  host_num=${#host_list[@]}
fi

hosts=("${host_list[@]:0:${host_num}}")
echo "Working on hosts:${hosts[@]}"

##############################################
#1 prepare oneflow_temp folder on each host
for host in "${hosts[@]}"
do
  ssh $USER@$host "mkdir -p ~/oneflow_temp"
done

##############################################
#2 copy files to each host and start work
for host in "${hosts[@]}"
do
  echo "start training on ${host}"
  ssh $USER@$host 'rm -rf ~/oneflow_temp/*'
  scp -r ./$BENCH_ROOT_DIR ./$LOCAL_RUN $USER@$host:~/oneflow_temp
  ssh $USER@$host "cd ~/oneflow_temp; nohup ./$LOCAL_RUN $NUM_NODES $GPU_NUM_PER_NODE 1>oneflow.log 2>&1 </dev/null &"
done
```

Note: Please to make sure all servers can login each other automaticly with ssh-key.

### Test Command Example
```
# test on 1 node with 4 gpus
./launch_all.sh 1 4

# test on 4 nodes with 8 gpus per node
./launch_all.sh 4 8
```

### Calculate `Throughput` from Test Results
`Throughput(samples/s)` information as well as `loss` and `top-k` can be found in `oneflow_temp` folder in the first node's home directory, there is a log file:
- `oneflow.log` - redirected stdout 

We use `oneflow.log` for instance, here is an example:
```
train: epoch 0, iter 20, loss: 6.505637, top_1: 0.000000, top_k: 0.000000, samples/s: 288.088
train: epoch 0, iter 40, loss: 5.736447, top_1: 0.020313, top_k: 0.117578, samples/s: 385.628
train: epoch 0, iter 60, loss: 4.274485, top_1: 0.817969, top_k: 0.991797, samples/s: 386.264
train: epoch 0, iter 80, loss: 2.331075, top_1: 1.000000, top_k: 1.000000, samples/s: 385.723
train: epoch 0, iter 100, loss: 1.236110, top_1: 1.000000, top_k: 1.000000, samples/s: 384.622
train: epoch 0, iter 120, loss: 1.078446, top_1: 1.000000, top_k: 1.000000, samples/s: 385.367
train: epoch 0, iter 140, loss: 1.054016, top_1: 1.000000, top_k: 1.000000, samples/s: 384.704
train: epoch 0, iter 160, loss: 1.048110, top_1: 1.000000, top_k: 1.000000, samples/s: 384.927
train: epoch 0, iter 180, loss: 1.050786, top_1: 1.000000, top_k: 1.000000, samples/s: 384.109
train: epoch 0, iter 200, loss: 1.047857, top_1: 1.000000, top_k: 1.000000, samples/s: 384.517
```
Normally, the first `samples/s` value e.g. `288.088` is discarded because the start time of first batch is not correct. we average the other `samples/s` as the throughput of this test.
## Test Results
All test logs can be found [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OF_benchmark_logs/oneflow_resnet50_logs.tgz)
### Group: batch size per device = 128
ResNet50 V1.5, batch size per device=128, dtype=float32, without XLA						
| node num | gpus/nodes | gpu num | bsz/gpu | GPU Memory Usage | Throughput | Speedup | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 1 | 128 | 12565 | 383.760  | 1 | 
| 1 | 2 | 2 | 128 | 12839 | 747.295  | 1.95  | 
| 1 | 4 | 4 | 128 | 12987 | 1497.618  | 3.90  | 
| 1 | 8 | 8 | 128 | 13051 | 2942.321  | 7.67  | 
| 2 | 8 | 16 | 128 | 12871 | 5839.054  | 15.22  | 
| 4 | 8 | 32 | 128 | 12871 | 11548.451  | 30.09  | 

### Group: batch size per device = 160
ResNet50 V1.5, batch size per device=160, dtype=float32, without XLA						
| node num | gpus/nodes | gpu num | bsz/gpu | GPU Memory Usage | Throughput | Speedup | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 1 | 160 | 15509 | 382.324  | 1 | 
| 1 | 2 | 2 | 160 | 15785 | 755.956  | 1.98  | 
| 1 | 4 | 4 | 160 | 15881 | 1494.733  | 3.91  | 
| 1 | 8 | 8 | 160 | 15701 | 3016.431  | 7.89  | 
| 2 | 8 | 16 | 160 | 15817 | 5877.289  | 15.37  | 
| 4 | 8 | 32 | 160 | 15879 | 11623.889  | 30.40  | 

