#! /bin/bash

GPU_NUM_PER_NODE=${1}
NODE_NUM=${2}
NODE_LIST=${3}
RUN_REAL_DATA=${4}
RUN_SYNTHETIC_DATA=${5}
LOGFILE=${6}

DATA_DIR=${OF_BERT_DATA_DIR}

SHELL_FOLDER=$(dirname $(readlink -f "$0"))

# bert base
CMD="python3 $SHELL_FOLDER/../bert_benchmark/run_pretraining.py \
--gpu_num_per_node=$GPU_NUM_PER_NODE \
--node_num=$NODE_NUM \
--node_list=$NODE_LIST \
--learning_rate=1e-4 \
--weight_l2=0.01 \
--batch_size_per_device=24 \
--iter_num=5 \
--loss_print_every_n_iter=1 \
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
--data_part_num=32 \
--data_dir=$DATA_DIR \
--log_dir=$OUTPUT_DIR/oneflow/bert/log \
--model_save_dir=$OUTPUT_DIR/oneflow/bert/model"

# bert large
#CMD="python3 $SHELL_FOLDER/../bert_benchmark/run_pretraining.py \
#--gpu_num_per_node=$GPU_NUM_PER_NODE \
#--node_num=$NODE_NUM \
#--node_list=$NODE_LIST \
#--learning_rate=1e-4 \
#--weight_l2=0.01 \
#--batch_size_per_device=24 \
#--iter_num=5 \
#--loss_print_every_n_iter=1 \
#--seq_length=512 \
#--max_predictions_per_seq=80 \
#--num_hidden_layers=24 \
#--num_attention_heads=16 \
#--max_position_embeddings=512 \
#--type_vocab_size=2 \
#--vocab_size=30522 \
#--attention_probs_dropout_prob=0.1 \
#--hidden_dropout_prob=0.1 \
#--hidden_size_per_head=64 \
#--data_part_num=32 \
# --data_dir=$DATA_DIR \
# --log_dir=$OUTPUT_DIR/oneflow/bert/log \
# --model_save_dir=$OUTPUT_DIR/oneflow/bert/model"


$CMD | tee ${LOGFILE}.log
echo "Saving log to ${LOGFILE}.log"


