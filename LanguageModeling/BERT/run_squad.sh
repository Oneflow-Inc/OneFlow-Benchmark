BENCH_ROOT_DIR=/path/to/
# pretrained model dir
PRETRAINED_MODEL=/DATA/disk1/of_output/uncased_L-12_H-768_A-12_oneflow

# squad ofrecord dataset dir
DATA_ROOT=/DATA/disk1/of_output/bert/of_squad

# `vocab.txt` dir
REF_ROOT_DIR=/DATA/disk1/of_output/uncased_L-12_H-768_A-12

# `evaluate-v*.py` and `dev-v*.json` dir
SQUAD_TOOL_DIR=/DATA/disk1/of_output/bert/of_squad
db_version=${1:-"v2.0"}
if [ $db_version = "v1.1" ]; then
  train_example_num=88614
  eval_example_num=10833
  version_2_with_negative="False"
elif [ $db_version = "v2.0" ]; then
  train_example_num=131944
  eval_example_num=12232
  version_2_with_negative="True"
else
  echo "db_version must be 'v1.1' or 'v2.0'"
  exit
fi

train_data_dir=$DATA_ROOT/train-$db_version
eval_data_dir=$DATA_ROOT/dev-$db_version
LOGFILE=./bert_fp_training.log
export PYTHONUNBUFFERED=1
export ONEFLOW_DEBUG_MODE=True
export CUDA_VISIBLE_DEVICES=7
# finetune and eval SQuAD,
# `predictions.json` will be saved to folder `./squad_output`
python3 $BENCH_ROOT_DIR/run_squad.py \
  --model=SQuAD \
  --do_train=True \
  --do_eval=True \
  --gpu_num_per_node=1 \
  --learning_rate=3e-5 \
  --batch_size_per_device=16 \
  --eval_batch_size_per_device=16 \
  --num_epoch=3 \
  --use_fp16 \
  --version_2_with_negative=$version_2_with_negative \
  --loss_print_every_n_iter=20 \
  --do_lower_case=True \
  --seq_length=384 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --train_data_dir=$train_data_dir \
  --train_example_num=$train_example_num \
  --eval_data_dir=$eval_data_dir \
  --eval_example_num=$eval_example_num \
  --log_dir=./log \
  --model_load_dir=${PRETRAINED_MODEL} \
  --save_last_snapshot=True \
  --model_save_dir=./squad_snapshots \
  --vocab_file=$REF_ROOT_DIR/vocab.txt \
  --predict_file=$SQUAD_TOOL_DIR/dev-${db_version}.json \
  --output_dir=./squad_output 2>&1 | tee ${LOGFILE}


# evaluate predictions.json to get metrics
python3 $SQUAD_TOOL_DIR/evaluate-${db_version}.py \
  $SQUAD_TOOL_DIR/dev-${db_version}.json \
  ./squad_output/predictions.json

