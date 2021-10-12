## Meta Knowledge Distillation
Oneflow实现[Meta Knowledge Distillation（Meta-KD）](https://arxiv.org/pdf/2012.01266.pdf")算法

---

## Meta-KD概述：
预训练语言模型进行知识蒸馏可以在尽可能不降低模型效果的前提下大大提高模型的执行效率，先前的方法则是遵循teacher-student方法实现模型蒸馏，但他们忽略了teacher模型学习得到的领域知识与student模型之间的偏差问题。基于此，Meta-KD提出采用元学习的方法，让teacher模型先学习得到不同domain上的meta knowledge，得到meta-teacher，然后再让student模型学习meta-teacher的先验知识，试图让student模型也具备meta-knowledge。

以文本分类任务为例，算法的流程大致如下：
- 首先获得N-way K-shot分类数据，根据每个数据集，获得各个类的prototypical embedding，并计算各个样本的prototypical score；
- 训练meta-teacher，采用元学习的方法学习domain-knowledge，包括prototype以及domain corruption等；
- 根据训练好的meta-teacher，在训练集上进行推理，得到每个样本的先验知识，包括该样本对应的attention values、最末层的表示向量以及预测的logits values；
- 训练meta-student，对于logits values则采用交叉信息熵损失函数，对于attention values和表示向量则采用平均MSE；

## 数据获取
以亚马逊评论评测任务为例，可下载语料：
The Amazon Review dataset can be found in this [link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

## 实验设置

#### Step1：生成prototype embedding并计算prototypical score

```shell
python3 preprocess.py \
        --task_name senti \
        --model_load_dir uncased_L-12_H-768_A-12_oneflow \
        --data_dir data/SENTI/ \
        --num_epochs 4 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --resave_ofrecord
```

执行完将得到相应的ofrecord数据


#### Step2：meta-teacher learning部分

本部分需要使用teacher模型在目标数据上进行元学习：

```shell
python3 meta_teacher.py \
        --task_name senti \
        --model_load_dir uncased_L-12_H-768_A-12_oneflow \
        --data_dir data/SENTI/ \
        --num_epochs 63 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --eval_example_num 720 \
        --batch_size_per_device 24 \
        --eval_batch_size_per_device 48 \
        --eval_every_step_num 100 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --learning_rate 5e-5 \
        --resave_ofrecord \
        --do_train \
        --do_eval
```
训练完后将得到meta-teacher模型，模型保存在output目录下

#### Step3：meta distillation部分

首先在训练集上，获得meta-teacher的soft-label、attention、embedding等参数，并保存至本地

```shell
python3 meta_teacher_eval.py \
        --task_name senti \
        --model_load_dir output/model_save-2021-09-26-15:31:15/snapshot_best_mft_model_senti_dev_0.8691358024691358 \ # 需要换为实际的目录
        --data_dir data/SENTI/ \
        --num_epochs 4 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --eval_batch_size_per_device 1 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --resave_ofrecord
```

然后再训练student模型，执行distillation：

```shell
python3 meta_distill.py 
        --task_name senti \
        --student_model uncased_L-12_H-768_A-12_oneflow \
        --teacher_model output/model_save-2021-09-26-15:31:15/snapshot_best_mft_model_senti_dev_0.8691358024691358 \ # 需要换为实际的目录
        --data_dir data/SENTI/ \
        --num_epochs 63 \
        --seed 42 \
        --seq_length=128 \
        --train_example_num 6480 \
        --eval_example_num 720 \
        --batch_size_per_device 24 \
        --eval_batch_size_per_device 48 \
        --eval_every_step_num 100 \
        --vocab_file uncased_L-12_H-768_A-12/vocab.txt \
        --learning_rate 5e-5 \
        --resave_ofrecord \
        --do_train \
        --do_eval
```
