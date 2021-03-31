Oneflow-Module版BERT实现，origin form:https://github.com/codertimo/BERT-pytorch

### dataset

示例数据集下载：[data.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/BERT-pytorch/sampledataset/data.zip) 并解压至本文件夹根目录

### requirements

oneflow版本为：https://github.com/Oneflow-Inc/oneflow/tree/dev_bert_module_test

同时，由于数据集加载部分的功能尚未对齐，需要依赖torch.utils.data.DataLoader，故需要依赖torch==0.4.1.post2

### test demo

`bash test.sh`