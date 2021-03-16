BENCH_ROOT_DIR=/path/to/OneFlow-Benchmark/LanguageModeling/BERT
OUTPUT_DIR=/DATA/disk1/of_output

DATA_DIR=/DATA/disk1/bert/wiki_seq_len_128


BZ=48
ITER_NUM=50000
max_seq_length=128
max_predictions_per_seq=20

of_log_dir=$OUTPUT_DIR/bert_master/of
rm -rf ${of_log_dir}
mkdir -p ${of_log_dir}
rm -rf core.*

export PYTHONUNBUFFERED=1
export ONEFLOW_DEBUG_MODE=True
export GLOG_v=3

python3 $BENCH_ROOT_DIR/run_pretraining.py \
  --gpu_num_per_node=8 \
  --num_nodes=1 \
  --learning_rate=1e-4 \
  --warmup_proportion=0.01 \
  --weight_decay_rate=0.01 \
  --batch_size_per_device=${BZ} \
  --iter_num=${ITER_NUM} \
  --loss_print_every_n_iter=1 \
  --seq_length=128 \
  --use_fp16 \
  --optimizer_type="lamb" \
  --max_predictions_per_seq=20 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.0 \
  --hidden_dropout_prob=0.0 \
  --hidden_size_per_head=64 \
  --data_part_num=64 \
  --data_dir=$DATA_DIR \
  --log_dir=${of_log_dir} \
  --model_save_every_n_iter=50000 \
  --model_save_dir=${of_log_dir}
