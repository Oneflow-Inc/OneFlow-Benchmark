
# 安装

我们在 OneFlow [#75f11b8](https://github.com/Oneflow-Inc/oneflow/commit/75f11b8257112c7afd0c777abf7cddc01b6b495c) 上进行了 OneFlow GPT 相关的测试。使用的 python 版本 3.7, cuda 11.2, cudnn 8.1.1, nccl 2.8.3。

为了节省用户自己编译打包 OneFlow 的时间，我们提供了已打包好的 [OneFlow #75f11b8 wheel](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/package/oneflow-0.3.5%2Bcu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl) 包。下载安装命令如下：

```
cd OneFlow-Benchmark/LanguageModeling/GPT
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/package/oneflow-0.3.5%2Bcu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl
python3 -m pip install oneflow-0.3.5+cu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl
python3 -m pip install -e .
```

为了节省用户配置环境的时间，我们提供了 [oneflow-manylinux2014-cuda11.2](https://oneflow-static.oss-cn-beijing.aliyuncs.com/docker_images/oneflow-manylinux2014-cuda11.2-0.1.tar.gz) 镜像供容器方式使用。镜像通过以下命令下载和加载：

```
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/docker_images/oneflow-manylinux2014-cuda11.2-0.1.tar.gz
docker load -i oneflow-manylinux2014-cuda11.2-0.1.tar.gz
```

以上镜像中并没有预先安装 OneFlow 的包，可以使用 tools 中的 [launch_container.py](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/GPT/tools/launch_container.py) 脚本来启动容器并启动预训练任务。在容器环境中训练 OneFlow GPT 时，不需要预先用 pip 来安装 OneFlow 和 OneFlow GPT，而是需要把 OneFlow wheel 包路径和 OneFlow GPT 源码路径作为参数传入 launch_container.py 脚本，后面会详细介绍。

# 训练

## 数据预处理

训练数据需要预处理。目前 OneFlow 数据预处理的过程和使用脚本参考 [Megatron-LM GPT](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)。

首先，将训练数据json格式保存，json的每行包含一个文本样本，数据放在`text`字段中，例如：
```
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```
可以使用[`preprocess_data.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py)中的`--json-key`标志更改json文件`text`的名称，其他字段是可选的，并且在处理过程中不使用。

然后将`json`文件处理为二进制格式以进行训练，处理命令如下：
```
python3 tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix gpt_sample_dataset \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
```
在这里，输出文件名为 `gpt_sample_dataset_text_document.bin` 和 `gpt_sample_dataset_text_document.idx`，是由 `--output-prefix` 参数加固定后缀 `_text_document.bin` 和 `_text_document.idx` 组合而成。

在后面会介绍的 GPT 的训练中，将会把不带扩展名的文件路径 `path/to/gpt_sample_dataset_text_document` 作为 `--dataset` 参数传入作为训练用数据集。

参数介绍：
- `--input`: 输入文件，即前面介绍的预处理好的`json`文件
- `--output-prefix`: 输出文件前缀
- `--vocab`: 词表文件
- `--dataset-impl`: 数据集格式，可设置为`mmap`，`cached`或`lazy`（默认为`mmap`）
- `--tokenizer-type`: token解析器类型
- `--merge-file`: BPE分词所需的merge文件
- `--append-eod`: 添加文件结束标志

进一步的命令行参数在源文件[`preprocess_data.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py)中描述。

词表文件 [gpt2-vocab.json](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) 和 BPE 分词所需 merge 文件 [gpt2-merges.txt](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) 可以直接下载。

我们测试使用的语料文件 openwebtext.json 是下载 [openwebtext.tar.xz](https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx) 后，然后通过工具转换而来。由于文件过大，暂不提供下载地址。

> 注：我们使用的语料文件只供测试使用，只是简单生成 json 格式语料文件，没有做过滤，清洗，去重等工作。

preprocess_data.py 的结果我们提供了直接下载的方式 [gpt_sample_dataset_text_document.bin](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/data/gpt_sample_dataset_text_document.bin), [gpt_sample_dataset_text_document.idx](https://oneflow-public.oss-cn-beijing.aliyuncs.com/GPT/data/gpt_sample_dataset_text_document.idx)。


## GPT 预训练

```
bash example/pretrain_345M.sh
```

这个脚本启动一个在单 GPU 上的拥有 345M 大小参数的 GPT 模型预训练任务。如果用户测试所使用的的 GPU 的显存比较小，也可以尝试使用更小参数的模型版本 `pretrain_117M.sh`。

用户如果想调整某些参数，也可以手动调用 [training.py](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/GPT/oneflow_gpt/training.py):

```
python3 -m oneflow_gpt.training \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --num-gpus-per-node 1 \
    --train-iters 500000 \
    --learning-rate 0.00015 \
    --dataset gpt_sample_dataset_text_document \
    --seq-length 1024 \
    --split 949,50,1 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-fraction 0.01 \
    --weight-decay 1e-2 \
    --optimizer adamw \
    --clip-grad 1.0 \
    --vocab-size 50257 \
    --load checkpoint \
    --save checkpoint \
    --save-interval 10000 \
    --log-interval 100 \
    --checkpoint-activations \
    --multihead-attention-fusion \
    --fp16
```

## 容器启动预训练

```
python3 tools/launch_container.py \
    --src $PWD \
    --py 3.7 \
    --image oneflow-manylinux2014-cuda11.2:0.1 \
    --wheel oneflow-0.3.5+cu112.git.75f11b825-cp37-cp37m-manylinux2014_x86_64.whl \
    --cmd "python3 -m oneflow_gpt.training \
                --dataset gpt_sample_dataset_text_document \
                --seq-length 1024 \
                --num-layers 24 \
                --hidden-size 1024 \
                --num-attention-heads 16 \
                --micro-batch-size 4 \
                --global-batch-size 8 \
                --tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --num-gpus-per-node 1 \
                --train-iters 500000 \
                --learning-rate 0.00015 \
                --min-lr 1.0e-5 \
                --lr-decay-style cosine \
                --lr-decay-iters 320000 \
                --lr-warmup-fraction 0.01 \
                --initial-loss-scale 4294967296 \
                --optimizer adamw \
                --weight-decay 1e-2 \
                --clip-grad 1.0 \
                --vocab-size 50257 \
                --split 949,50,1 \
                --load checkpoint \
                --save checkpoint \
                --save-interval 10000 \
                --log-interval 100 \
                --checkpoint-activations \
                --multihead-attention-fusion \
                --fp16"
```

可以利用 `tools/launch_container.py` 脚本来在容器中执行 GPT 的预训练任务。我们需要提供源码位置（`--src`）、容器镜像（`--image`）、python 版本（`--py`）、还有 OneFlow wheel 包位置（`--wheel`）等参数，最后还要通过 `--cmd` 参数来传递我们想要执行的 GPT Pretrain 的脚本命令。

## 分布式预训练

当我们想训练更大的隐层（`--hidden-size`），更深层次的网络时(`--num-layers`)，往往要利用并行训练的技术。这个时候我们需要关注以下重要的参数:

- `--tensor-model-parallel-size`: 模型并行数
- `--pipeline-model-parallel-size`: 流水并行数

配合以上参数，结合之前已经使用过的 `--num-gpus-per-node`、`--num-nodes` 就可以推导出并行训练的全貌。其中还包含2个隐式的参数（不用手动配置，由其他参数推导）:

- `world_size`: 全部的设备数，等于 `--num-gpus-per-node` * `--num-nodes`
- `data_parallel_size`: 数据并行数。等于 `world_size` // `--tensor-model-parallel-size` // `--pipeline-model-parallel-size`

同时了为了达到更好的训练效果，我们一般还会配置梯度累加（`--num-accumulation-steps`）。流水并行时一般需要配置梯度累加才能达到较好的效果。`--num-accumulation-steps` 与 `--micro-batch-size` 和 `--global-batch-size` 存在以下关系：

`--global-batch-size` == `--micro-batch-size` * `data_parallel_size` * `--num-accumulation-steps`

如果我们是在单机多卡上运行并行训练，则启动方式与之前的单设备并无差异。以下是单机8卡上，数据和模型混合并行的示例（不带流水并行）:

```
bash examples/pretrain_1n8d_2x4x1_16_1536x16.sh
```

如果需要进一步扩充设备数量，由于单台机器适配的 GPU 设备数量有限，我们需要更多的物理机器 node。此时，我们需要配置 `--num-nodes` 和 `--node-ips` 参数，并且分别在每台机器上启动训练命令（`traning.py`）。同时在有 rdma 的环境中，可以开启 `--use-rdma` 来带来更佳的训练效率。以下是4机8卡下，各种并行方式混合的示例：

```
bash examples/distribute_pretrain_4n8d_2x4x4_512_2304x24.sh
```

更多的参数配置见 [config.py](https://github.com/NVIDIA/Megatron-LM/blob/main/oneflow_gpt/config.py)

# 模型评估和下游任务 (Evaluation and Tasks)

## LAMBADA Cloze Accuracy

[WIP]
