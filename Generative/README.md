# GAN 生成对抗网络演示



## 简介 Introduction

生成对抗网络（GAN）任务的本质是学习一个数据分布。它包含生成网络和判别网络两部分。其中生成网络可将一个随机分布映射为任意分布，判别网络则决定了生成分布的“方向”。二者相互博弈的过程在理论上等效于分布的拟合过程。本文演示了DCGAN的运行，DCGAN是一种广泛应用于图像生成领域中的基于卷积/反卷积运算的生成对抗网络，


## 演示 Demo

可通过脚本直接启动DCGAN的训练：

```bash
python dcgan.py
```

脚本参数：

- `-lr` 学习率，默认为1e-4
-  `-e` 设置训练epoch次数，默认为10
- `-b` 设置batchsize
- `-g` 设置gpu数量



其他需要注意的是：

- 训练将会默认使用minst数据集，如果第一次使用脚本，将会默认将数据集下载到`.data/`目录
- 训练结束后，将会默认保存模型到`.checkpoint/`目录下
- 模型的结构和参数参考了tensorflow的[官方示例](https://www.tensorflow.org/tutorials/generative/dcgan)
- 模型会定期将生成的图片存储到`.gout/`目录，并在训练结束后生成图片演化的动图,生成动图的过程会依赖python包`imageio`

![dcgan demo](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/dev_gan/Generative/pic/1.png)
