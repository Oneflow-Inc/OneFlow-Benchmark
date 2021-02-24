#!/bin/bash
export PYTHONUNBUFFERED=1
model=${1:-gpt2-small}
batch_size_per_device=${2:-1}
node_num=${3:-1}
gpu_num_per_node=${4:-1}
checkpoint_activations=${5:-"off"}
enable_non_distributed_optimizer=${6:-"False"}
dtype=${7:-"fp16"}
iter_num=${8:-200}
model_paraller_size=${9:-1}

if  [ ${model} = "gpt2-small" ];then
    echo "network : gpt2-small"
    n_layer=12
    n_head=12
    n_embd=768
elif  [ ${model} = "gpt2-medium" ];then
    echo "network : gpt2-medium"
	n_layer=24
    n_head=16
    n_embd=1024
fi

script_path=$(realpath $0)
gpt_root=$(dirname $script_path)
dataset=$gpt_root/data/wiki_00
cfg_dir=$gpt_root/models/117M

total_batch_size=`expr ${batch_size_per_device} \* ${gpu_num_per_node}`
echo "total_batch_size >>>> $total_batch_size"
seq_len=1024
dropout_rate=0.1

if  [ ${enable_non_distributed_optimizer} = "True" ];then
    flag=1
else
    flag=0
fi

output_dir=20210224-stage-${flag}-${model}-${checkpoint_activations}-checkpoint-activations/${node_num}n${gpu_num_per_node}g
test_case=${model}_b${batch_size_per_device}_${dtype}_1
mkdir -p $output_dir
mem_file=$output_dir/$test_case.mem
log_file=$output_dir/$test_case.log

python3 $gpt_root/tools/gpu_memory_usage.py -g=$gpu_num_per_node -n=1 >$mem_file 2>&1 </dev/null &

cmd=""
# cmd+="gdb --args "
# cmd+="/opt/nvidia/nsight-systems-2020.5.1/bin/nsys profile -o $test_case "
cmd+="python3 -m src.train "
cmd+="--dataset=$dataset "
cmd+="--cfg_dir=$cfg_dir "
cmd+="--iter_num=$iter_num "
cmd+="--loss_print_every_n_iter=10 "
cmd+="--seq_len=$seq_len "
cmd+="--n_vocab=50272 "
cmd+="--n_ctx=1024 "
cmd+="--n_embd=$n_embd "
cmd+="--n_head=$n_head "
cmd+="--n_layer=$n_layer "
cmd+="--optimizer=adam "
cmd+="--embedding_dropout=$dropout_rate "
cmd+="--hidden_dropout=$dropout_rate "
cmd+="--attention_dropout=$dropout_rate "

if [ "$dtype" = "fp16" ] ; then
	echo "using dtype : fp16"
    cmd+="--use_fp16=True "
else
    echo "using dtype : fp32"
    cmd+="--use_fp16=False "
fi

if  [ ${checkpoint_activations} = "on" ];then
    echo "checkpoint_activations on"
    cmd+="--checkpoint_activations "
else
    echo "checkpoint_activations off"
fi

if  [ ${enable_non_distributed_optimizer} = "True" ];then
    cmd+="--enable_non_distributed_optimizer=True "
else
    cmd+="--enable_non_distributed_optimizer=False "
fi


cmd+="--use_big_fc=False "
cmd+="--metric_print_format=table "
cmd+="--total_batch_size=$total_batch_size "
cmd+="--gpu_num_per_node=$gpu_num_per_node "
cmd+="--num_nodes=$node_num "
cmd+="--node_ips=10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5 "

echo "CMD: $cmd"

echo
    date +%s.%N
    $cmd 2>&1 | tee ${log_file}
    date +%s.%N



