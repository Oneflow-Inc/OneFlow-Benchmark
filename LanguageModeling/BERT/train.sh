#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
DATA_DIR=/dataset/bert_regression_test/0
python3 run_pretraining.py \
  --gpu_num_per_node=1 \
  --learning_rate=1e-4 \
  --batch_size_per_device=32 \
  --iter_num=1000 \
  --loss_print_every_n_iter=20 \
  --seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.0 \
  --hidden_dropout_prob=0.0 \
  --hidden_size_per_head=64 \
  --data_part_num=1 \
  --data_dir=$DATA_DIR \
  --log_dir=./log \
  --model_save_every_n_iter=10000 \
  --save_last_snapshot=True \
  --model_save_dir=./snapshots 
  # --model_load_dir=/dataset/bert_regression_test/of_random_init_L-12_H-768_A-12