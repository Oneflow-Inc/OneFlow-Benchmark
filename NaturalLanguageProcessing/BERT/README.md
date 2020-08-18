BERT - Bidirectional Encoder Representations from Transformers

OneFlow实现了BERT的预训练模型（Pre-training）和两个NLP下游任务（SQuAD，Classifier）。主要参考了[谷歌](https://github.com/google-research/bert)和[英伟达](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)的实现。

本文的主要目的是介绍如何快速的使用OneFlow BERT相关脚本。

## 数据集下载
在使用这些脚本进行训练、精调或预测之前，请参考下面链接下载相应的数据集，数据集的格式是OFRecord。
- Pretrain数据集：完整的预训练数据集是由[Wikipedia](https://dumps.wikimedia.org/)和[BookCorpus](http://yknzhu.wixsite.com/mbweb)两部分数据制作而成，制作成OFRecord之后大概200G。我们提供了一个[样本数据集点击下载](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/wiki_ofrecord_seq_len_128_example.tgz)，仅供测试使用；
- SQuAD数据集：包括完整数据集和相关工具，[下载地址](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/squad_dataset_tools.tgz)，解压目录为`squad_dataset_tools`，包括如下文件：
```
squad_dataset_tools
├── ofrecord 
├── dev-v1.1.json  
├── dev-v2.0.json  
├── train-v1.1.json  
├── train-v2.0.json
├── evaluate-v1.1.py  
├── evaluate-v2.0.py  
```
- GLUE(CoLA, MRPC)：完整数据集[下载地址](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/glue_ofrecord.tgz)，解压目录为`glue_ofrecord`，包括如下文件：
```shell
glue_ofrecord
├── CoLA
│   ├── eval
│   │   └── eval.of_record-0
│   ├── test
│   │   └── predict.of_record-0
│   └── train
│       └── train.of_record-0
└── MRPC
    ├── eval
    │   └── eval.of_record-0
    ├── test
    │   └── predict.of_record-0
    └── train
        └── train.of_record-0
```

如果感兴趣，可以通过[google-research BERT](https://github.com/google-research/bert)提供的工具脚本，制作tfrecord格式的数据集。再根据[加载与准备OFRecord数据集](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/cn/docs/extended_topics/how_to_make_ofdataset.md)中的方法，将TFRecord数据转为OFRecord数据集使用。

## 用BERT进行预训练，Pre-training with BERT
用BERT进行预训练，除了预训练相关的python脚本之外，您只需要准备好数据集，并设置`data_dir`为该数据集的目录，下面就是预训练python脚本运行的示例脚本：
```bash
DATA_DIR=/path/to/wiki_ofrecord_seq_len_128_example
python3 run_pretraining.py \
  --gpu_num_per_node=1 \
  --learning_rate=1e-4 \
  --batch_size_per_device=64 \
  --iter_num=1000000 \
  --loss_print_every_n_iter=20 \
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
  --data_part_num=1 \
  --data_dir=$DATA_DIR \
  --log_dir=./log \
  --model_save_every_n_iter=10000 \
  --save_last_snapshot=True \
  --model_save_dir=./snapshots
```
修改上面脚本中的`DATA_DIR`后运行，屏幕上首先会打印出参数列表，接着如果能看到类似下面的输出，就说明BERT预训练任务已经成功的开始运行了。
```
step: 19, total_loss: 11.138, mlm_loss: 10.422, nsp_loss: 0.716, throughput: 81.638
step: 39, total_loss: 10.903, mlm_loss: 10.216, nsp_loss: 0.687, throughput: 121.513
step: 59, total_loss: 10.615, mlm_loss: 9.922, nsp_loss: 0.692, throughput: 104.633
step: 79, total_loss: 10.347, mlm_loss: 9.655, nsp_loss: 0.692, throughput: 118.725
```
需要注意的是`model_save_dir`指明的是保存模型的目录，如果该目录存在，运行会报错，请先删除该目录，OneFlow在运行时会自动创建目录。

还需要注意的一个参数是：`data_part_num`。这个参数指明了数据集中数据文件（part）的个数。我们提供的示例数据集只有一个part，所以设置为1，如果您有多个part的数据，请根据实际的数量配置。

## Using BERT in SQuAD
### Step 0: 输入和模型准备
如果需要完整的精调SQuAD网络，需要准备下面一些文件：
1. 数据集
2. 预训练好的BERT模型
3. 词表文件`vocab.txt`
4. SQuAD官方的评估工具，`evaluate-v1.1.py`或`evaluate-v2.0.py`
5. SQuAD原始评估数据集文件，`dev-v1.1.json`或`dev-v2.0.json`

其中1、4、5我们提供了下载链接，4和5在SQuAD官网也能够下载。
下面介绍如何准备OneFlow需要的预训练好的模型，词表文件也包含在其中的下载文件里。
#### 将Tensorflow的BERT模型转为OneFlow模型格式
如果想直接使用已经训练好的pretrained模型做fine-tune任务（如以下将展示的SQuAD），可以考虑直接从[google-research BERT](https://github.com/google-research/bert)页面下载已经训练好的BERT模型。

再利用我们提供的`convert_tf_ckpt_to_of.py`脚本，将其转为OneFlow模型格式。转换过程如下：

首先，下载并解压某个版本的BERT模型，如`uncased_L-12_H-768_A-12`。
```shell
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d uncased_L-12_H-768_A-12
```

然后，运行以下命令：
```shell
cd uncased_L-12_H-768_A-12/
cat > checkpoint <<ONEFLOW
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
ONEFLOW
```

该命令将在解压目录下创建一个`checkpoint`文件，并写入以下内容：
```
model_checkpoint_path: "bert_model.ckpt" 
all_model_checkpoint_paths: "bert_model.ckpt" 
```

此时，已经准备好待转化的tensorflow模型目录，整个模型目录的结构如下：
```shell
uncased_L-12_H-768_A-12
├── bert_config.json
├── bert_model.ckpt.data-00000-of-00001
├── bert_model.ckpt.index
├── checkpoint
└── vocab.txt
```

我们接着使用`convert_tf_ckpt_to_of.py`将tensorflow模型转为OneFlow模型：
```bash
python convert_tf_ckpt_to_of.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12 \
  --of_dump_path ./uncased_L-12_H-768_A-12_oneflow
```
以上命令，将转化好的OneFlow格式的模型保存在`./uncased_L-12_H-768_A-12_oneflow`目录下，供后续微调训练(如SQuAD)使用。

### Step 1: 精调并验证
通过运行`run_squad.py`进行SQuAd的精调并生成用于评测打分的文件`predictions.json`，然后调用`evaluate-v*.py`工具进行评测打分，运行脚本如下。要运行下面的脚本，您需要根据实际情况配置一下相关路径。
```bash
# pretrained model dir
PRETRAINED_MODEL=/path/to/uncased_L-12_H-768_A-12_oneflow

# squad ofrecord dataset dir
DATA_ROOT=/path/to/squad_dataset_tools/ofrecord

# `vocab.txt` dir
REF_ROOT_DIR=/path/to/uncased_L-12_H-768_A-12

# `evaluate-v*.py` and `dev-v*.json` dir
SQUAD_TOOL_DIR=/path/to/squad_dataset_tools

db_version=v1.1
if [ $db_version = "v1.1" ]; then
  train_example_num=88614
  eval_example_num=10833
elif [ $db_version = "v2.0" ]; then
  train_example_num=131944
  eval_example_num=12232
else
  echo "db_version must be 'v1.1' or 'v2.0'"
  exit
fi

train_data_dir=$DATA_ROOT/train-$db_version
eval_data_dir=$DATA_ROOT/dev-$db_version

# finetune and eval SQuAD, 
# `predictions.json` will be saved to folder `./squad_output`
python3 run_squad.py \
  --model=SQuAD \
  --do_train=True \
  --do_eval=True \
  --gpu_num_per_node=1 \
  --learning_rate=3e-5 \
  --batch_size_per_device=16 \
  --eval_batch_size_per_device=16 \
  --num_epoch=2 \
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
  --output_dir=./squad_output

# evaluate predictions.json to get metrics
python3 $SQUAD_TOOL_DIR/evaluate-${db_version}.py \
  $SQUAD_TOOL_DIR/dev-${db_version}.json \
  ./squad_output/predictions.json
```
根据您使用的设备不同，整个过程可能要花费几十分钟到几个小时的时间，最终会打印出类似如下的结果：
```shell
{"exact_match": 81.46641438032167, "f1": 88.7223407359181}
```

## BERT Classification with CoLA and MRPC
这个任务只需要配置预训练好的模型目录和数据集目录即可运行，调用脚本是`run_classifier.py`，调用方法参考下面的脚本：
```bash
# pretrained model dir
PRETRAINED_MODEL=/path/to/uncased_L-12_H-768_A-12_oneflow

# ofrecord dataset dir
DATA_ROOT=/path/to/glue_ofrecord

# choose dateset `CoLA` or `MRPC`
dataset=CoLA
#dataset=MRPC
if [ $dataset = "CoLA" ]; then
  train_example_num=8551
  eval_example_num=1043
  test_example_num=1063
  learning_rate=1e-5
  wd=0.01
elif [ $dataset = "MRPC" ]; then
  train_example_num=3668
  eval_example_num=408
  test_example_num=1725
  learning_rate=2e-6
  wd=0.001
else
  echo "dataset must be 'CoLA' or 'MRPC'"
  exit
fi

train_data_dir=$DATA_ROOT/${dataset}/train
eval_data_dir=$DATA_ROOT/${dataset}/eval

python3 run_classifier.py \
  --model=Glue_$dataset \
  --task_name=$dataset  \
  --gpu_num_per_node=1 \
  --num_epochs=4 \
  --train_data_dir=$train_data_dir \
  --train_example_num=$train_example_num \
  --eval_data_dir=$eval_data_dir \
  --eval_example_num=$eval_example_num \
  --model_load_dir=${PRETRAINED_MODEL} \
  --batch_size_per_device=32 \
  --eval_batch_size_per_device=4 \
  --loss_print_every_n_iter 20 \
  --log_dir=./log \
  --model_save_dir=./snapshots \
  --save_last_snapshot=True \
  --seq_length=128 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --learning_rate $learning_rate \
  --weight_decay_rate $wd
```
上面的代码中，可以通过设置`dataset`来选择使用`CoLA`还是`MRPC`数据集。

正常运行时除了会打印loss等基本信息之外，每训练完一个epoch，都会对训练集和验证集进行一次完整的预测，并打印出一些指标结果，如下：
```shell
train accuarcy: 0.917, matthews_corrcoef: 0.797, precision: 0.906, recall: 0.984, f1: 0.943
eval accuarcy: 0.781, matthews_corrcoef: 0.445, precision: 0.788, recall: 0.934, f1: 0.855
```