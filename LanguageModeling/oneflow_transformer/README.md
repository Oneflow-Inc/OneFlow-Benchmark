# TransFormer 模型搭建

## 代码目录

### model2 
以Mlperf的tensorflow官方实现进行对齐

部分代码未添加，因为缺失算子

## translate_ende 
用于存放词表

## utils
### metric.py 
包含一些损失函数计算，评价指标
## tokenizer.py 
分词器实现
## tokenizer_test.py
分词器简单测试

# 数据准备
可以运行 data_download_preprocess.py 文件

该代码完成以下事项
1. 原始数据集的下载 raw_data 
2. 创建分词器，以及对应的词表
3. 分别对训练集，评价集进行分词，分块。
4. 单独对训练集每一个Block做一次shuffle，并以OFRecord形式存储
相关保存路径均可以在parser修改

# 现有问题
缺失两个算子OP，正在开发。