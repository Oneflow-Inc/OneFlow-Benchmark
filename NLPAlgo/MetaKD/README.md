The Amazon Review dataset can be found in this [link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

生成prototype embedding并计算prototypical score

python3 preprocess.py --task_name senti --model_load_dir uncased_L-12_H-768_A-12_oneflow --data_dir data/SENTI/ --num_epochs 4 --seed 42 --seq_length=128 --train_example_num 6480 --vocab_file uncased_L-12_H-768_A-12/vocab.txt --resave_ofrecord

meta-teacher learning部分

python3 meta_teacher.py --task_name senti --model_load_dir uncased_L-12_H-768_A-12_oneflow --data_dir data/SENTI/ --num_epochs 63 --seed 42 --seq_length=128 --train_example_num 6480 --eval_example_num 720 --batch_size_per_device 24 --eval_batch_size_per_device 48 --eval_every_step_num 100 --vocab_file uncased_L-12_H-768_A-12/vocab.txt --learning_rate 5e-5 --resave_ofrecord --do_train --do_eval

meta distillation部分

首先在训练集上，获得meta-teacher的soft-label、attention、embedding等参数，并保存至本地
python3 meta_teacher_eval.py --task_name senti --model_load_dir output/model_save-2021-09-26-15:31:15/snapshot_best_mft_model_senti_dev_0.8691358024691358 --data_dir data/SENTI/ --num_epochs 4 --seed 42 --seq_length=128 --train_example_num 6480 --eval_batch_size_per_device 1 --vocab_file uncased_L-12_H-768_A-12/vocab.txt --resave_ofrecord


执行distillation
python3 meta_distill.py --task_name senti --student_model uncased_L-12_H-768_A-12_oneflow --teacher_model output/model_save-2021-09-26-15:31:15/snapshot_best_mft_model_senti_dev_0.8691358024691358 --data_dir data/SENTI/ --num_epochs 63 --seed 42 --seq_length=128 --train_example_num 6480 --eval_example_num 720 --batch_size_per_device 24 --eval_batch_size_per_device 48 --eval_every_step_num 100 --vocab_file uncased_L-12_H-768_A-12/vocab.txt --learning_rate 5e-5 --resave_ofrecord --do_train --do_eval
