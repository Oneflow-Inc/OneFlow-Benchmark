# OneFlow BERT Pretrain Benchmark Test Report
This document reports OneFlow BERT Pretrain benchmark test results on Aug 13 2020. 

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
4 groups of tests were performed with different batch size per device: 32, 64 and 96 for BERT base, 4 for BERT large.

Each group includes 6 tests with different number of devices: 1, 2, 4, 8, 16, 32.

`Throughput` of images/sec and `GPU Memory Usage` were logged and recorded.

Data type of all tests is `Float32`, XLA is not applied.

## Test Scripts
Please clone or download `BERT` folder from [OneFlow-Benchmark repository](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling/BERT). 

We create two bash scripts alone side with `BERT` folder for this test:
1. `local_run.sh` - launch a local oneflow with specific number of nodes and gpu number per node
```bash
# local_run.sh
NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BENCH_ROOT_DIR=BERT

DATA_ROOT=/path/to/ofrecord
rm -rf ./log
mkdir ./log

#BSZ_PER_DEVICE=32
#BSZ_PER_DEVICE=64
BSZ_PER_DEVICE=96

python3 ./$BENCH_ROOT_DIR/run_pretraining.py \
  --gpu_num_per_node=$GPU_NUM_PER_NODE \
  --num_nodes=$NUM_NODES \
  --node_ips='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5' \
  --learning_rate=1e-4 \
  --batch_size_per_device=$BSZ_PER_DEVICE \
  --iter_num=200 \
  --loss_print_every_n_iter=20 \
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
  --data_dir=$DATA_ROOT \
  --data_part_num=32 \
  --log_dir=./log \
  --model_save_every_n_iter=10000 \
  --save_last_snapshot=False \
  --model_save_dir=./snapshots

```
2. `launch_all.sh` - launch oneflow on all remote nodes with specific number of nodes and gpu number per node.
```bash
# launch_all.sh
#!/bin/bash

NUM_NODES=$1
GPU_NUM_PER_NODE=$2
LOCAL_RUN=local_run.sh
BENCH_ROOT_DIR=BERT

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
`Throughput(samples/s)` information as well as `loss` can be found in `oneflow_temp` folder in the first node's home directory, there are two files:
1. `oneflow.log` - redirected stdout 
2. `log/summary.csv` - same information in csv format 

We use `oneflow.log` for instance, here is an example:
```
step: 19, total_loss: 11.078, mlm_loss: 10.407, nsp_loss: 0.671, throughput: 52.257
step: 39, total_loss: 10.884, mlm_loss: 10.190, nsp_loss: 0.694, throughput: 142.735
step: 59, total_loss: 10.592, mlm_loss: 9.915, nsp_loss: 0.677, throughput: 142.636
step: 79, total_loss: 10.335, mlm_loss: 9.659, nsp_loss: 0.676, throughput: 142.391
step: 99, total_loss: 10.157, mlm_loss: 9.479, nsp_loss: 0.678, throughput: 142.565
step: 119, total_loss: 10.046, mlm_loss: 9.361, nsp_loss: 0.686, throughput: 142.397
step: 139, total_loss: 9.915, mlm_loss: 9.237, nsp_loss: 0.678, throughput: 142.298
step: 159, total_loss: 9.851, mlm_loss: 9.168, nsp_loss: 0.683, throughput: 142.383
step: 179, total_loss: 9.784, mlm_loss: 9.104, nsp_loss: 0.680, throughput: 142.270
step: 199, total_loss: 9.640, mlm_loss: 8.960, nsp_loss: 0.680, throughput: 142.579
```
Normally, the first `throughput` value e.g. `52.257` is discarded because the start time of first batch is not correct. we average the other `throughput` as the throughput of this test.
## BERT base Pretrain Test Results
All test logs can be found [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OF_benchmark_logs/oneflow_bert_benchmark_logs.tgz)
### Group: batch size per device = 32
BERT Base Pretrain, batch size per device=32, dtype=float32, without XLA						
| node num | device num | bsz per device | throughput | speedup | memory(MiB) | 
| -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 32 | 137.17  | 1.00  | 6205 | 
| 1 | 2 | 32 | 250.41  | 1.83  | 7071 | 
| 1 | 4 | 32 | 502.70  | 3.66  | 7139 | 
| 1 | 8 | 32 | 990.87  | 7.22  | 7215 | 
| 2 | 16 | 32 | 1573.31  | 11.47  | 7135 | 
| 4 | 32 | 32 | 3081.96  | 22.47  | 7149 | 

### Group: batch size per device = 64 
BERT Base Pretrain, batch size per device=64, dtype=float32, without XLA						
| node num | device num | bsz per device | throughput | speedup | memory(MiB) | 
| -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 64 | 145.55  | 1.00  | 9987 | 
| 1 | 2 | 64 | 277.03  | 1.90  | 10847 | 
| 1 | 4 | 64 | 551.78  | 3.79  | 10923 | 
| 1 | 8 | 64 | 1105.13  | 7.59  | 11057 | 
| 2 | 16 | 64 | 2016.09  | 13.85  | 10937 | 
| 4 | 32 | 64 | 3911.90  | 26.88  | 10963 | 


### Group: batch size per device = 96 
BERT Base Pretrain, batch size per device=96, dtype=float32, without XLA						
| node num | device num | bsz per device | throughput | speedup | memory(MiB) | 
| -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 96 | 148.34  | 1.00  | 13769 | 
| 1 | 2 | 96 | 286.24  | 1.93  | 14735 | 
| 1 | 4 | 96 | 573.85  | 3.87  | 14809 | 
| 1 | 8 | 96 | 1147.47  | 7.74  | 14893 | 
| 2 | 16 | 96 | 2169.65  | 14.63  | 14763 | 
| 4 | 32 | 96 | 4238.85  | 28.58  | 14795 | 

## BERT Large Pretrain Test Results
BERT large was tested on the same situtation. Some arguments in `local_run.sh` need to be modified to meet to BERT large pretrain configuration. 
```bash
# local_run.sh for bert large
NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BENCH_ROOT_DIR=BERT

DATA_ROOT=/path/to/ofrecord
rm -rf ./log
mkdir ./log

BSZ_PER_DEVICE=4

python3 ./$BENCH_ROOT_DIR/run_pretraining.py \
  --gpu_num_per_node=$GPU_NUM_PER_NODE \
  --num_nodes=$NUM_NODES \
  --node_ips='10.11.0.2','10.11.0.3','10.11.0.4','10.11.0.5' \
  --learning_rate=1e-4 \
  --batch_size_per_device=$BSZ_PER_DEVICE \
  --iter_num=200 \
  --loss_print_every_n_iter=20 \
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
  --data_dir=$DATA_ROOT \
  --data_part_num=32 \
  --log_dir=./log \
  --model_save_every_n_iter=10000 \
  --save_last_snapshot=False \
  --model_save_dir=./snapshots

```
Here is the result:
BERT Large Pretrain, batch size per device=4, dtype=float32, without XLA
| node num | gpu num/node | gpu num | bsz/gpu | GPU Memory Usage | Throughput | Speedup | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| 1 | 1 | 1 | 4 | 12087 | 8.839  | 1 | 
| 1 | 2 | 2 | 4 | 14593 | 16.405  | 1.86  | 
| 1 | 4 | 4 | 4 | 14713 | 33.158  | 3.75  | 
| 1 | 8 | 8 | 4 | 14765 | 64.519  | 7.30  | 
| 2 | 8 | 16 | 4 | 14661 | 74.224  | 8.40  | 
| 4 | 8 | 32 | 4 | 14673 | 143.232  | 16.21  | 
| 1 | 1 | 1 | 6 | 15779 | 9.180  | 1.04  | 

