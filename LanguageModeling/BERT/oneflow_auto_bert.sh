#!/bin/bash

BENCH_ROOT=$1
PYTHON_WHL=$2
CMP_OLD=$3

# PYTHON_WHL=oneflow-0.3.5+cu112.git.325160b-cp38-cp38-linux_x86_64.whl
# CMP_OLD=325160bcfb786b166b063e669aea345fadee2da7

BERT_OSSDIR=oss://oneflow-staging/branch/master/bert/
DOWN_FILE="wget https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/bert/${CMP_OLD}/out.tar.gz"
# DOWN_FILE="ossutil64 cp ${BERT_OSSDIR}$CMP_OLD/out.tar.gz .; "
ENABLE_FP32=0
GPU_NUM_PER_NODE=8
BSZ=64

PORT=57520

PYTHON="python3.8"
DOCKER_USER=root

multi_machine() 
{
    # param 1 node
    NUM_NODES=$1

    # param 2 run cmd
    RUN_CMD=$2

    # param 3 output file
    OUTPUT_FILE=$3

    # param 4 python 
    PYTHON=$4

    # param 5  
    IS_F32=$5

    declare -a host_list=("10.11.0.2" "10.11.0.3" "11.10.0.4" "10.11.0.5")

    if [ $NUM_NODES -gt ${#host_list[@]} ]
    then
        echo num_nodes should be less than or equal to length of host_list.
        exit
    fi

    hosts=("${host_list[@]:0:${NUM_NODES}}")
    echo "Working on hosts:${hosts[@]}"

    ips=${hosts[0]}
    for host in "${hosts[@]:1}" 
    do
        ips+=",${host}"
    done

    for host in "${hosts[@]:1}" 
    do
    echo "start training on ${host}"

    echo -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
        nohup $RUN_CMD 0 $PYTHON  >/dev/null 2>&1 &" 

    ssh -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
        nohup $RUN_CMD 0 $PYTHON  >/dev/null 2>&1 &"   

    done

    # copy files to master host and start work
    host=${hosts[0]}
    echo "start training on ${host}"

    echo $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
        $RUN_CMD 1 $PYTHON "
    ssh -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
        $RUN_CMD 1 $PYTHON "


    for host in "${hosts[@]}" 
    do
        echo $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
            mkdir -p out/${OUTPUT_FILE}; mv -f log out/${OUTPUT_FILE}/log_1 "    
        ssh -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
            mkdir -p out/${OUTPUT_FILE}; mv -f log out/${OUTPUT_FILE}/log_1 "
    done

    # Result analysis

    host=${hosts[0]}
    echo "start training on ${host}"

    echo -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
        $PYTHON tools/result_analysis.py $IS_F32 \
        --cmp1_file=./old/$OUTPUT_FILE/log_1/out.json \
        --cmp2_file=./out/$OUTPUT_FILE/log_1/out.json \
        --out=./pic/$OUTPUT_FILE.png "


    ssh -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
        $PYTHON tools/result_analysis.py $IS_F32 \
        --cmp1_file=./old/$OUTPUT_FILE/log_1/out.json \
        --cmp2_file=./out/$OUTPUT_FILE/log_1/out.json \
        --out=./pic/$OUTPUT_FILE.png "

    echo "multi_machine done"

}


#######################################################################################
#                     0 prepare the host list ips for training
########################################################################################
ALL_NODES=4

declare -a host_list=("10.11.0.2" "10.11.0.3" "10.11.0.4" "10.11.0.5")

if [ $ALL_NODES -gt ${#host_list[@]} ]
then 
    echo num_nodes should be less than or equal to length of host_list.
    exit
fi

hosts=("${host_list[@]:0:${ALL_NODES}}")
echo "Working on hosts:${hosts[@]}"

ips=${hosts[0]}
for host in "${hosts[@]:1}" 
do
   ips+=",${host}"
done

# #######################################################################################
# #                     1 prepare oneflow_temp folder on each host
# ########################################################################################

for host in "${hosts[@]}" 
do
    ssh -p $PORT $DOCKER_USER@$host " rm -rf ~/oneflow_temp ; mkdir -p ~/oneflow_temp"
    scp -P $PORT -r $BENCH_ROOT  $DOCKER_USER@$host:~/oneflow_temp/
    echo "tesst--->"
    scp -P $PORT -r $PYTHON_WHL  $DOCKER_USER@$host:~/oneflow_temp/
    ssh -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/; \
       $PYTHON -m pip install $PYTHON_WHL; "

    ssh -p $PORT $DOCKER_USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
                                    mkdir -p pic; rm -rf pic/*; mkdir -p out; rm -rf out/* "
    

done

#_______________________________________________________________________________________________
host=${hosts[0]}
ssh -p $PORT $DOCKER_USER@$host "cd ~; rm -rf ~/out; \
                                ${DOWN_FILE}; \
                                tar xvf out.tar.gz; \
                                cp -rf ~/out ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT/old;"


#######################################################################################
#                                2   run   single
########################################################################################

NUM_NODES=1


if [ "$ENABLE_FP32" = 1 ];then
    
    #                                f32  adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 1 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f32_pretraining_8gpu_${BSZ}bs_130iter_adam_debug" \
        $PYTHON "--f32=1"
    #                                f32  lamb debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 1 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f32_pretraining_8gpu_${BSZ}bs_100iter_lamb_debug" \
        $PYTHON "--f32=1"
    #                                f32 accumulation adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 2 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f32_pretraining_8gpu_${BSZ}bs_100iter_accumulation_debug" \
        $PYTHON "--f32=1"
    #                                f32 accumulation lamb debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 2 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f32_pretraining_8gpu_${BSZ}bs_100iter_accumulation_lamb_debug" \
        $PYTHON "--f32=1"
    echo "BERT USE_FP32"
else
    #                                f16  adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 1 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_debug" \
        $PYTHON "--f32=0"
    #                                f16  lamb debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 1 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_lamb_debug" \
        $PYTHON "--f32=0"

    #                                f16 accumulation adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 2 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_accumulation_debug" \
        $PYTHON "--f32=0"
    #                                f16 accumulation lamb
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 2 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "single_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_accumulation_lamb_debug" \
        $PYTHON "--f32=0"
    echo "BERT USE_FP16"

fi




#____________________________________________________________________________________________________



# #######################################################################################
# #                              2   run multi-machine
# ########################################################################################
NUM_NODES=4


if [ "$ENABLE_FP32" = 1 ];then
    
    #                                f32  adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 1 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f32_pretraining_8gpu_${BSZ}bs_130iter_adam_debug" \
        $PYTHON "--f32=1"
    #                                f32  lamb debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 1 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f32_pretraining_8gpu_${BSZ}bs_100iter_lamb_debug" \
        $PYTHON "--f32=1"
    #                                f32 accumulation adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 2 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f32_pretraining_8gpu_${BSZ}bs_100iter_accumulation_debug" \
        $PYTHON "--f32=1"
    #                                f32 accumulation lamb debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 0 1 ${BSZ} 2 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f32_pretraining_8gpu_${BSZ}bs_100iter_accumulation_lamb_debug" \
        $PYTHON "--f32=1"

    echo "BERT USE_FP32"
else
    #                                f16  adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 1 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_debug" \
        $PYTHON "--f32=0"
    #                                f16  lamb debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 1 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_lamb_debug" \
        $PYTHON "--f32=0"
    #                                f16 accumulation adam debug
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 2 adam ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_accumulation_debug" \
        $PYTHON "--f32=0"
    #                                f16 accumulation lamb
    multi_machine ${NUM_NODES} "sh train_perbert.sh 1 1 ${BSZ} 2 lamb ${GPU_NUM_PER_NODE} $NUM_NODES " \
        "multi_bert_f16_pretraining_8gpu_${BSZ}bs_100iter_accumulation_lamb_debug" \
        $PYTHON "--f32=0"
    echo "BERT USE_FP16"
fi


# __________________________________________________________________________________________

host=${hosts[0]}
echo "start tar on ${host}"

ssh $USER@$host "cd ~/oneflow_temp/OneFlow-Benchmark/LanguageModeling/BERT; \
    tar  -zcvf out.tar.gz  out; \
    $PYTHON  tools/stitching_pic.py --dir=pic --out_file=./pic/all.png "

echo "multi_machine done"
