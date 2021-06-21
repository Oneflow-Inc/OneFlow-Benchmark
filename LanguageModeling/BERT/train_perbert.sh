USE_FP16=${1:-0}
#DEBUG
DEBUG_MODE=${2:-0}
#
BATCH_SIZE=${3:-64}
#accumulation
ACCUMULATION_STEMPS=${4-1}
#
OPTIMIZER=${5-adam}
#GPU
GPUS_PER_NODE=${6:-8}
#
NNODES=${7:-1}
#
MASTER=${8:-0}

PYTHON=${9:-python}

LOG_FOLDER=./log/

PRINT_ITER=1
ITER_NUM=130

NODE_IPS='10.10.0.2','10.10.0.3','10.10.0.4','10.10.0.5'

# INIT_MODEL=/opt/initial_model 
INIT_MODEL=/data/bert/initial_model/
#DATA_DIR=/data/bert_dataset
DATA_DIR=/data/bert/wiki_seq_len_128/

##########################################################################################################
#                                           FP
##########################################################################################################
echo ${USE_FP16}


if [ "$USE_FP16" = 1 ];then
  FP_CMD=--use_fp16
  FP_NAME=f16
  echo "USE_FP16"
else
  FP_CMD=
  FP_NAME=f32
  echo "USE_FP32"
fi
##########################################################################################################
#                                           DEBUG_NAME
##########################################################################################################
# if [ DEBUG_MODE==1 ];then
#   DEBUG_NAME=debug
# else
#   DEBUG_NAME=
# fi

##########################################################################################################
#                                           Create folder
##########################################################################################################
#bert_f32_pretraining_8gpu_64bs_100iter_lamb_debug
mkdir -p $LOG_FOLDER

# OUTFILE=bert_pretraining_${FP_NAME}_${GPUS_PER_NODE}gpu_${BATCH_SIZE}bs_${ITER_NUM}iter_${OPTIMIZER}\
# _${DEBUG_NAME}
# mkdir -p $OUTFILE

LOGFILE=$LOG_FOLDER/bert_pretraining_${FP_NAME}_${GPUS_PER_NODE}gpu_${BATCH_SIZE}bs_${ITER_NUM}iter_${OPTIMIZER}\
_${DEBUG_NAME}.log

MODEL_DIR=./snapshots/

MEM_FILE=$LOG_FOLDER/memory.log

echo LOGFILE=$LOGFILE
echo DATA_DIR=$DATA_DIR

#${NNODES}n${GPUS_PER_NODE}g_dp${D_P}_mp${M_P}_pp${P_P}_mbs${MICRO_BATCH_SIZE}_gbs${GLOABAL_BATCH_SIZE}_pretrain_${NODE_RANK}.log
rm -rf ${MODEL_DIR}/*
rm -rf ${LOG_FOLDER}/*

NVPROF=baseline-report_${NODE_RANK}
#-o ${NVPROF} 


# -g $GPUS_PER_NODE \
# -n 0.5 \


export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1


#nsys profile --stats=true -o ${NVPROF}  \

$PYTHON run_pretraining.py \
  --gpu_num_per_node=${GPUS_PER_NODE} \
  --num_nodes=${NNODES} \
  --node_ips=$NODE_IPS \
  --learning_rate=1e-4 \
  --warmup_proportion=0.01 \
  --weight_decay_rate=0.01 \
  --batch_size_per_device=${BATCH_SIZE} \
  --iter_num=${ITER_NUM} \
  --loss_print_every_n_iter=${PRINT_ITER} \
  --seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --num_accumulation_steps=${ACCUMULATION_STEMPS} \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0 \
  --hidden_dropout_prob=0 \
  --hidden_size_per_head=64 \
  --data_part_num=64 \
  --data_dir=$DATA_DIR \
  --log_dir=${LOG_FOLDER} \
  --model_save_every_n_iter=10000 \
  --save_last_snapshot=True \
  --model_save_dir=./snapshots \
  --debug=${DEBUG_MODE} \
  --data_load_random=0 \
  --model_load=${INIT_MODEL} \
  ${FP_CMD} \
  --optimizer_type=${OPTIMIZER} \
  2>&1 | tee ${LOGFILE} 

echo "Writting log to ${LOGFILE}"

SQLITE=$LOG_FOLDER/bert_pretraining_${GPUS_PER_NODE}gpu_${BATCH_SIZE}bs_${ITER_NUM}iter.sqlite
QDREP=$LOG_FOLDER/bert_pretraining_${GPUS_PER_NODE}gpu_${BATCH_SIZE}bs_${ITER_NUM}iter.qdrep

# mv $NVPROF.sqlite $SQLITE
# mv $NVPROF.qdrep  $QDREP

if [ "$MASTER" = 1 ]; then
  json_file=${LOG_FOLDER}out.json
  python tools/analysis.py \
    --log_file=$LOGFILE \
    --mem_file=$MEM_FILE \
    --out_file=$json_file \
    --gpu_num=$GPUS_PER_NODE 

fi
