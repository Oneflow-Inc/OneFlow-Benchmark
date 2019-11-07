#! /bin/bash

GPU_NUM_PER_NODE=${1}
NODE_NUM=${2}
NODE_LIST=${3}
RUN_REAL_DATA=${4}
RUN_SYNTHETIC_DATA=${5}
LOGFILE=${6}

DATA_DIR=${OF_CNN_DATA_DIR}

SHELL_FOLDER=$(dirname $(readlink -f "$0"))

CMD="python3 $SHELL_FOLDER/../cnn_benchmark/of_cnn_benchmarks.py \
--model=vgg16 \
--gpu_num_per_node=$GPU_NUM_PER_NODE \
--node_num=$NODE_NUM \
--node_list=$NODE_LIST \
--batch_size_per_device=64 \
--iter_num=100 \
--learning_rate=0.128 \
--optimizer=sgd \
--loss_print_every_n_iter=1 \
--warmup_iter_num=20 \
--data_part_num=16 \
--log_dir=$OUTPUT_DIR/oneflow/cnns/log \
--model_save_dir=$OUTPUT_DIR/oneflow/cnns/model"

# synthetic data
if [ $RUN_SYNTHETIC_DATA = "True" ] ; then
    ${CMD} | tee ${LOGFILE}_sythetic.log
    echo "Saving log to ${LOGFILE}_sythetic.log"
fi

# real data
if [ $RUN_REAL_DATA = "True" ]; then
    CMD+=" --data_dir=$DATA_DIR"
    ${CMD} | tee ${LOGFILE}_real.log
    echo "Saving log to ${LOGFILE}_real.log"
fi


