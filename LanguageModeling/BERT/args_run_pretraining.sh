#! /bin/bash
# set -ex

# bash args_run_pretraining.sh ${NUM_NODES} ${NUM_GPUS_PER_NODE} ${BSZ_PER_DEVICE} ${USE_FP16} ${ITER_NUM} ${LOSS_PRINT_ITER} ${DATA_DIR} ${DATA_PART_NUM} ${SEQ_LENGHT} ${NUM_HIDDEN_LAYERS} ${NUM_ATTENTION_HEADS} ${PYTHON_BIN} ${NODE_IPS} ${NSYS_BIN} ${RUN_COMMIT}

NUM_NODES=${1:-1}
NUM_GPUS_PER_NODE=${2:-8}
BSZ_PER_DEVICE=${3:-48}
USE_FP16=${4:-true}
ITER_NUM=${5:-300}
LOSS_PRINT_ITER=${6:-10}
DATA_DIR=${7:-""}
# leinao:/data/bert/wiki_seq_len_128
# jinshan: /DATA/disk1/bert/wiki_seq_len_128
DATA_PART_NUM=${8:-64}
SEQ_LENGHT=${9:-128} # large:512
NUM_HIDDEN_LAYERS=${10:-12} # large:24
NUM_ATTENTION_HEADS=${11:-12} # large: 16
PYTHON_BIN=${12:-"python3"}
NODE_IPS=${13:-"10.11.0.2,10.11.0.3,10.11.0.4,10.11.0.5"}
DEBUG_AND_NCCL=${14:-false}
NSYS_BIN=${15:-""}
RUN_COMMIT=${16:-"master"}
NUM_ACC_STEP=${17:-1}
OPTIMIZER_TYPE=${18:-"adam"}

SRC_DIR=$(realpath $(dirname $0))

RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=$SRC_DIR/test_logs/$HOSTNAME/${NUM_NODES}n${NUM_GPUS_PER_NODE}g
mkdir -p $LOG_FOLDER
LOG_FILENAME=$LOG_FOLDER/bert_${NUM_NODES}n${NUM_GPUS_PER_NODE}g_sq${SEQ_LENGHT}_nhl${NUM_HIDDEN_LAYERS}_nah${NUM_ATTENTION_HEADS}_bsz${BSZ_PER_DEVICE}_${OPTIMIZER_TYPE}_${RUN_COMMIT}_${RUN_TIME}

export PYTHONUNBUFFERED=1
export GLOG_v=3
export NCCL_LAUNCH_MODE=GROUP

echo DEBUG_AND_NCCL=$DEBUG_AND_NCCL
if $DEBUG_AND_NCCL; then
    export ONEFLOW_DEBUG_MODE=1
    echo ONEFLOW_DEBUG_MODE=$ONEFLOW_DEBUG_MODE
    export NCCL_DEBUG=INFO
    echo NCCL_DEBUG=$NCCL_DEBUG
fi

if [[ ${NUM_NODES} -gt 1 ]]; then
    export ONEFLOW_COMM_NET_IB_ENABLE=1
fi

CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    CMD+="${NSYS_BIN} profile --stats true --output ${LOG_FILENAME} "
fi

CMD+="${PYTHON_BIN} run_pretraining.py "
CMD+="--gpu_num_per_node=${NUM_GPUS_PER_NODE} "
CMD+="--num_nodes=${NUM_NODES} "
CMD+="--node_ips=${NODE_IPS} "
CMD+="--learning_rate=1e-4 "
CMD+="--warmup_proportion=0.01 "
CMD+="--weight_decay_rate=0.01 "
CMD+="--batch_size_per_device=${BSZ_PER_DEVICE} "
CMD+="--iter_num=${ITER_NUM} "
CMD+="--loss_print_every_n_iter=${LOSS_PRINT_ITER} "
CMD+="--seq_length=${SEQ_LENGHT} "
if $USE_FP16; then
    echo USE_FP16=$USE_FP16
    CMD+="--use_fp16 "
fi
CMD+="--num_accumulation_steps=${NUM_ACC_STEP} "
CMD+="--optimizer_type=${OPTIMIZER_TYPE} "
CMD+="--max_predictions_per_seq=20 "
CMD+="--num_hidden_layers=${NUM_HIDDEN_LAYERS} "
CMD+="--num_attention_heads=${NUM_ATTENTION_HEADS} "
CMD+="--max_position_embeddings=512 "
CMD+="--type_vocab_size=2 "
CMD+="--vocab_size=30522 "
#CMD+="--attention_probs_dropout_prob=0.1 "
if [[ ${ITER_NUM} -gt 800 ]]; then
    CMD+="--attention_probs_dropout_prob=0.0 "
    CMD+="--hidden_dropout_prob=0.0 "
fi
#CMD+="--hidden_dropout_prob=0.0 "
CMD+="--hidden_size_per_head=64 "
CMD+="--data_part_num=${DATA_PART_NUM} "
CMD+="--data_dir=${DATA_DIR} "
#CMD+="--log_dir=${LOG_FOLDER} "
CMD+="--model_save_every_n_iter=10000 "
CMD+="--model_save_dir=${LOG_FOLDER} "

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

echo "Writting log to ${LOG_FILENAME}"

rm -rf ./log/$HOSTNAME
rm -rf ./output/$HOSTNAME
