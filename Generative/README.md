# GAN 生成对抗网络演示



## 简介 Introduction

生成对抗网络（GAN）任务的本质是学习一个数据分布。它包含生成网络和判别网络两部分。其中生成网络可将一个随机分布映射为任意分布，判别网络则决定了生成分布的“方向”。二者相互博弈的过程在理论上等效于分布的拟合过程。



本文包括了两种生成模型

- DCGAN：一种基于卷积/反卷积运算的生成对抗网络，广泛应用于图像生成领域中

- Pix2Pix：一种基于DCGAN的风格迁移网络，其基本思想是在保证基本图片纹理一致的情况下，通过GAN网络实现一对一的风格转化



## DCGAN

可通过脚本直接启动DCGAN的训练：

```bash
python dcgan.py
```

脚本参数：

- `-lr` 学习率，默认为1e-4
-  `-e` 设置训练epoch次数，默认为10
- `-c` 与tensorflow进行对齐测试
- `-b` 设置batchsize
- `-g` 设置gpu数量
- `-m` 多机训练



其他需要注意的是：

- 训练将会默认使用minst数据集，如果第一次使用脚本，将会默认将数据集下载到`.data/`目录
- 训练结束后，将会默认保存模型到`.checkpoint/`目录下
- 模型的结构和参数参考了tensorflow的[官方示例](https://www.tensorflow.org/tutorials/generative/dcgan)，可以通过-c参数来跟tensorflow的实现进行对齐测试
- 模型会定期将生成的图片存储到`.gout/`目录，并在训练结束后生成图片演化的动图

![](https://raw.githubusercontent.com/JamiePlur/picgo/master/20200615170256.png)

# Pix2Pix



可通过脚本直接启动Pix2Pix的训练：

```bash
python pix2pix.py
```

脚本参数：

- `-lr` 学习率，默认为1e-4
-  `-e` 设置训练epoch次数，默认为10
- `-c` 进行对齐测试
- `-b` 设置batchsize
- `-g` 设置gpu数量
- `-m` 多机训练



其他需要注意的是：

- 训练将会默认使用CMP Facade数据集，如果第一次使用脚本，将会默认将数据集下载到`.data/`目录
- 训练结束后，将会默认保存模型到`.checkpoint/`目录下
- 模型的结构和参数参考了tensorflow的[官方示例](https://www.tensorflow.org/tutorials/generative/pix2pix)，可以通过-c参数来跟tensorflow的实现进行对齐测试
- 模型会在训练中定期将生成的图片存储到`.gout/`目录

![image-20200701153752019](https://raw.githubusercontent.com/JamiePlur/picgo/master/20200701153829.png)

## [TODO]



### GAN精度评价

- GAN的精度评价只存在于基本gan结构下，风格迁移没有定量的精度评价
- GAN最主要的评价指标主要为inception score和Fréchet Inception Distance，都需要借助预训练的inceptionV3模型，并在imagenet分类数据集上使用，目前还没有实现
- 比较SOTA的GAN模型为了冲榜，都会附上性能指标
- 可以参考一些论文对GAN的评估工作，比较有代表性的是[How good is my gan？](https://lear.inrialpes.fr/people/alahari/papers/shmelkov18.pdf)

![image-20200701103940612](https://raw.githubusercontent.com/JamiePlur/picgo/master/20200701154135.png)



### GAN的预训练模型

- 目前还没有可用的预训练模型，自行训练的主要困难有：
  - GAN涉及的数据集非常多，数据处理的流水线没有建立起来
  - GAN的调参非常困难，稳定性差
- 只有在大型数据集上的预训练模型才有意义，训练mnist数据集这种模型没有泛化价值



### CycleGAN

- cyclegan已经搭建完成，但是由于模型较大，跑不起来
- gan的模型普遍较大