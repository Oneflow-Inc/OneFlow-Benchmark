#!/usr/bin/env bash

echo ========download the tensorflow model=======

#mkdir -p ./log/uncased_L-12_H-768_A-12
#wget -P ./log https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
#unzip -d ./log/uncased_L-12_H-768_A-12 ./log/uncased_L-12_H-768_A-12.zip
cp ./log/uncased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001 ./log/uncased_L-12_H-768_A-12/bert_model.ckpt

echo -e "\n"

echo ========build the oneflow model=======

mkdir -p ./log/of_uncased_L-12_H-768_A-12
python ./run_glue.py --data_dir glue_data_bin/CoLA/train --val_data_dir glue_data_bin/CoLA/eval \
    --save_and_break True \
    --model_save_dir ./log/of_uncased_L-12_H-768_A-12 \


echo -e "\n"

echo =======convert tensorflow model to oneflow model=======

python ./convert_tf_ckpt_to_of.py \
    --tf_checkpoint_path ./log/uncased_L-12_H-768_A-12/bert_model.ckpt \
    --of_dump_path ./log/of_uncased_L-12_H-768_A-12
