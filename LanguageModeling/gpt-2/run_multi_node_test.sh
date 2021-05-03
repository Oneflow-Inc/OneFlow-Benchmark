#!/usr/bin/bash
current_dir=$(dirname $(readlink -f "$0"))
model=${1:-gpt2-small}
batch_size_per_device=${2:-16}
zero_stage=${3:-0}
checkpoint_activations=${4:-"on"}
dtype=${5:-'fp16'}
test_num=${6:-5}


i=1
while [ $i -le ${test_num} ]
do
    bash $current_dir/test_zero_optimization.sh $model $batch_size_per_device 2 8  $checkpoint_activations $zero_stage $dtype ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done

i=1
while [ $i -le ${test_num} ]
do
    bash $current_dir/test_zero_optimization.sh $model $batch_size_per_device 4 8  $checkpoint_activations $zero_stage $dtype ${i}
    echo " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Finished Test Case ${i}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< "
    let i++
    sleep 20s
done
