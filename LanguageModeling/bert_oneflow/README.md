Oneflow-Module版BERT实现，origin form:https://github.com/codertimo/BERT-pytorch

### dataset

示例数据集下载：[data.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/BERT-pytorch/sampledataset/data.zip) 并解压至本文件夹根目录

### requirements

1.oneflow版本直接用master分支上的oneflow即可：https://github.com/Oneflow-Inc/oneflow/

2.同时，由于数据集加载部分的功能尚未对齐，需要依赖torch.utils.data.DataLoader，故依赖torch，需安装torch.

3.需手动将bert_pytorch.zip解压并复制到site-packages下，如：

`cp -r bert_pytorch ~/anaconda3/envs/oneflow/lib/python3.7/site-packages/`

### demo


- train: `bash train.sh`
- test: `bash test.sh`


