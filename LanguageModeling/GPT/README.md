# OneFlow GPT
## 数据预处理
训练数据需要预处理。目前OneFlow数据预处理的过程和使用脚本与[Megatron-LM GPT](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)一致。

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
       --output-prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
```
在这里，输出文件名为`my-gpt2_text_document.bin`和`my-gpt2_text_document.idx`。在GPT训练中，将使用不带扩展名的文件名称用作`--data-path`即`my-gpt2_text_document`。

参数介绍：
- `--input`: 输入文件，即前面介绍的预处理好的`json`文件
- `--output-prefix`: 输出文件前缀
- `--vocab`: 词表文件
- `dataset-impl`: 数据集格式，可设置为`mmap`，`cached`或`lazy`（默认为`mmap`）
- `--tokenizer-type`: token解析器类型
- `--merge-file`: BPE分词所需的merge文件
- `append-eod`: 添加文件结束标志

进一步的命令行参数在源文件[`preprocess_data.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py)中描述。
