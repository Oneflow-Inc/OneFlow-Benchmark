## 简介 Introduction

## 图像分类与CNN

**图像分类** 是指将图像信息中所反映的不同特征，把不同类别的目标区分开来的图像处理方法，是计算机视觉中其他任务，比如目标检测、语义分割、人脸识别等高层视觉任务的基础。

ImageNet大规模视觉识别挑战赛（ILSVRC），常称为ImageNet竞赛，包括图像分类、物体定位，以及物体检测等任务，是推动计算机视觉领域发展最重要的比赛之一。

在2012年的ImageNet竞赛中，深度卷积网络AlexNet横空出世。以超出第二名10%以上的top-5准确率，勇夺ImageNet2012比赛的冠军。从此，以 CNN（卷积神经网络） 为代表的深度学习方法开始在计算机视觉领域的应用开始大放异彩，更多的更深的CNN网络被提出，比如ImageNet2014比赛的冠军VGGNet, ImageNet2015比赛的冠军ResNet。

OneFlow-Benchmark下的cnn仓库目前已支持resnet50、vgg、alexnet等经典的cnn模型，未来会陆续添加新的cnn模型。这些cnn模型共享一套训练、验证和推理代码，您只需要指定模型，即可使用一套代码完成这些cnn网络模型的训练、测试和验证。

## 快速开始 Quick Start

### 准备工作 Requirements

别担心，使用OneFlow非常容易，只要准备好下面三步，即可开始OneFlow的图像识别之旅。

- 安装OneFlow。 

  - 直接通过pip安装：`pip install oneflow`  （TODO：确定我们的pip源是否做好,问caishenghang）
  - 其他安装方式：参考[这里](XXX)（TOOD：待补充链接，链接到编译安装的文档说明） 。

- 下载[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark)仓库。

  `git clone git@github.com:Oneflow-Inc/OneFlow-Benchmark.git`

- 准备数据集（可选）

  - 下载[示例数据集](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenet_ofrecord_example/part-00000`)

  - 或者：制作完整OFRecord格式的ImageNet数据集（见下文进阶部分）

  - 再或者：直接使用虚拟“合成数据”。

**关于数据集的说明：**


> 1）本文的展示的代码中，使用OFRcord格式的数据集可以提高数据加载效率（但这非必须，参考[数据输入](https://github.com/Oneflow-Inc/oneflow-documentation/docs/basics_topics/data_input.md)，oneflow支持直接加载numpy数据）。
>
> 2）为了使读者快速上手，我们提供了一个小的示例数据集。直接下载，即可快速开始训练过程。读者可以在熟悉了流程后，可以参考数据集制作部分，制作完整的数据集。
>
> 3）“合成数据”是指不通过磁盘加载数据，而是直接在内存中生成一些随机数据，作为网络的数据输入源。


### 训练&验证

```shell
sh train.sh
```

执行脚本开始模型的训练，每轮(epoch)训练后，将自动执行模型的验证。默认情况下，我们使用resnet50，您也可以通过改动脚本中的--model参数指定其他模型，如：--model="resnet50"，--model="vgg"等。

**参数说明**(部分)

- --train_data_dir                Imagenet2012训练集文件夹路径(ofrecord格式)

- --train_data_part_num   训练所用的ofrecord分片数量

- --val_data_dir                    Imagenet2012验证集文件夹路径(ofrecord格式)

- --val_data_part_num       验证所用的ofrecord分片数量

- --num_nodes=1                训练使用的机器节点数

- --gpu_num_per_node      每个机器节点使用的gpu数量

- --model_update="momentum"    学习率更新方式

- --learning_rate=0.256               初始学习率

- --loss_print_every_n_iter          打印loss间隔 

- --batch_size_per_device            训练时每个gpu的batch大小

- --val_batch_size_per_device     验证时每个gpu的batch大小

- --num_epoch                              迭代总轮数

- --model                                        使用的模型

### 预测/推理
```shell
sh inference.sh
```

**参数说明**(部分)

- --model 指定要加载的模型
- --image_path 待检测图片路径
- --model_load_dir 模型文件路径

以上，是针对cnns文件夹下所有CNN模型的训练、验证的通用脚本，下面，我们将重点介绍经典CNN网络：Resnet，以及如何利用OneFlow训练Resnet50，并对标Nvidia的Mxnet版实现。

## ResNet

[ResNet](https://arxiv.org/abs/1512.03385) 是2015年ImageNet竞赛的冠军。目前，ResNet相对对于传统的机器学习分类算法而言，效果已经相当的出色，之后大量的检测，分割，识别等任务也都在ResNet基础上完成。

[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark)仓库中，我们提供了ResNet50 v1.5的OneFlow实现。该实现对标了[英伟达的Mxnet版实现](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)。我们在ImageNet-2012数据集上训练90轮后，验证集上的准确率能够达到：77.318%(top1)，93.622%(top5)  更详细的网络参数对齐工作，见下面【进阶 Advanced】部分。

![resnet50_validation_acuracy](docs/resnet50_validation_acuracy.png)


**关于ResNet50 v1.5的说明：**

> ResNet50 v1.5是原始[ResNet50 v1](https://arxiv.org/abs/1512.03385)的一个改进版本，相对于原始的模型，精度稍有提升 (~0.5% top1)，详细说明参见[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) 。



准备好亲自动手，复现上面的结果了吗？那么接下来，立马开始OneFlow的图像识别之旅吧！

下面，本文就以上面的ResNet50 为例，一步步展现如何使用OneFlow进行网络的训练和预测。

### 训练和验证（Train & Validation）

训练开始前，需要提前准备好数据集，具体见上面的【准备工作 Requirements】部分，准备好之后就可以进行下面的步骤了。

先切换到代码目录：

```
cd OneFlow-Benchmark/Classification/cnns
```

设置train.sh脚本中的训练集/验证集的路径等参数，然后在命令行执行：

```
sh train.sh
```

若在屏幕上不断打印出类似下面的信息，则表明训练过程正常运行：

```
train: epoch 0, iter 200, loss: 7.024337, top_1: 0.000957, top_k: 0.005313, samples/s: 964.656
train: epoch 0, iter 400, loss: 6.849526, top_1: 0.003594, top_k: 0.012969, samples/s: 991.474
...
train: epoch 0, iter 5000, loss: 5.557458, top_1: 0.064590, top_k: 0.174648, samples/s: 935.390
Saving model to ./output/snapshots/model_save-20200629223546/snapshot_epoch_0.
validation: epoch 0, iter 100, top_1: 0.074620, top_k: 0.194120, samples/s: 2014.683
```

可以看到：

- 随着训练的进行，loss不断下降，而训练的top_1/top_k准确率不断提高（其中top_k默认为top_5准确率，可自定义）。
- 每个epoch结束时，会做另外两个工作：1）执行一次验证，并打印出验证集上的top_1/top_k准确率；2）保存模型。
- samples/s 用来指示训练/验证的执行速度，即每秒钟能处理的图片数量。



**复现实验的说明：**

> Q1. 多久能够完成训练？
>
> 在GPU环境下，使用单机8卡（NVIDIA TITAN V），完成90个epoch的完整训练过程，大概需要15小时。
>
> Q2. 在ImageNet-2012数据集上训练90个epoch后，准确率能达到多少？
>
> 训练集：80.57%（top1）
>
> 验证集：77.318%（top1），93.622%（top5）



### 预测（Inference）

恭喜，到这里，您已经知道如何用OneFlow训练模型，接下来，试试用训练好的模型对新图片进行分类预测吧！

在预测之前，需要准备你自己训练的模型，或者用我们在下面提供的在完整Imagenet(2012)上训练好的模型。准备好模型后，将模型目录填入`inference.sh` 脚本的`MODEL_LOAD_DIR`变量中，然后执行以下命令，开始对图片`test_img/tiger.jpg`的类别的进行预测：

```
sh inference.sh
```

若输出下面的内容，则表示预测成功：

```
test_img/tiger.jpg
0.81120294 tiger, Panthera tigris
```

### 模型（Model）

**关于模型，您可以选择：**

- 自己训练的模型

- 或者，下载我们已训练好的模型：[resnet_v1.5_model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_v15_of_best_model_val_top1_77318.tgz ) 

  (validation accuracy: 77.318% top1，93.622% top5 )
  
  


## 更详细的说明 Details

### 分布式训练
**简单而易用的分布式，是OneFlow的主打特色之一。**

OneFlow框架从底层设计上，就原生支持高效的分布式训练。尤其对于分布式的数据并行，用户完全不用操心算法从单机单卡扩展到多机多卡时，数据如何划分以及同步的问题。也就是说，使用OneFlow，用户以单机单卡的视角写好算法，**自动具备多机多卡分布式数据并行的能力。**


#### 如何配置并运行分布式训练？
还是以上面"快速开始"部分演示的代码为例，在`train.sh`中，只要用`--num_nodes` 指定节点（机器）个数，同时用`--node_ips`指定节点的ip地址，然后用`--gpu_num_per_node`指定每个节点上使用的卡数，就轻松地完成了分布式的配置。

例如，想要在2机8卡上进行分布式训练，像下面这样配置：

```shell
# train.sh 
python3 of_cnn_train_val.py \
    --num_nodes=2 \
    --node_ips="192.168.1.1, 192.168.1.2"
    --gpu_num_per_node=4 \
    ...
    --model="resnet50"
```

然后分别在两台机器上，同时执行：
```shell
./train.sh
```

程序启动后，通过`watch -n 0.1 nvidia-smi`命令可以看到，两台机器的GPU都开始了工作。一段时间后，会在`--node_ips`设置中的第一台机器的屏幕上，打印输出。


### 混合精度训练与预测

目前，OneFlow已经原生支持半精度/全精度的混合精度训练。训练时，模型参数（权重）使用float16进行训练，同时保留float32用作梯度更新和计算过程。由于参数的存储减半，会带来训练速度的提升。

在OneFlow中开启半精度/全精度的混合精度训练模式，ResNet50的训练速度理论上能达到`1.7`倍的加速。


#### 如何开启半精度/全精度混合精度训练？

只需要在`train.sh`脚本中添加参数`--use_fp16=True`即可。

#### 混合精度模型

我们为您提供了一个在Imagenet2012完整训练了90个epoch的混合精度模型，top_1：77.33%

您可以直接下载使用：[resnet50_v15_fp16](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_fp16_of_best_model_val_top1_77330.zip)



## 进阶 Advanced

### 参数对齐

Oneflow的ResNet50实现，为了保证和[英伟达的Mxnet版实现](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)对齐，我们从learning rate学习率，优化器Optimizer的选择，数据增强的图像参数设定，到更细的每一层网络的形态，bias,weight初始化等都做了细致且几乎完全一致的对齐工作。

#### Data Augmentation

**训练**

1. 随机采样图像并将其解码为[0; 255]。
2. 随机裁剪一个矩形区域，该矩形区域的长宽比以[3/4; 4/3]和以[8％；100％]，然后将裁剪的区域调整为224 x 224平方的图像。
3. 以0.5的概率水平翻转。
4. 色彩增强，比例色相，饱和度和亮度，其系数从[0.6; 1.4]。
5. 将PCA噪声与从正态分布N（0，0.1）采样的系数相加。
6. 通过分别减去123.68、116.779、103.939并除以58.393、57.12、57.375来标准化RGB通道。
7. 调整图像的大小，使其较短的一面在[256，480]中随机采样以进行缩放。随机抽取224×224区域。

| item                     | oneflow | nvidia |
| ------------------------ | ------- | ------ |
| 1 random sample          | Yes     | Yes    |
| 2 random crop resize     | Yes     | Yes    |
| 7 short side resize crop | No      | No     |
| 3 Flip horizontally      | Yes     | Yes    |
| 4 Color augmentation     | No      | No     |
| 5 PCA Noise              | No      | No     |
| 6.1 Normalize mean       | Yes     | Yes    |
| 6.2 Normalize std        | Yes     | Yes    |

**验证**

- 将每个图像的短边调整为256像素，同时保持其宽高比。
- 裁剪中心的224×224区域
- 标准化RGB通道，类似于训练。

#### Learning Rate Schedule

Oneflow保持了和Mxnet一致的初始学习率以及衰减方式。具体来说，我们采用了5个epoch的warmup，初始学习率lr = 0.256，lr衰减方式采用cosine decay（初始lr可根据batch_size和gpu数量可线性缩放）

- warmup + cosine decay
- warmup + step decay

[![image](docs/resnet50_lr_schedule.png)

| item        | oneflow | nvidia  |
| ----------- | ------- | ------ |
| start lr    | 0.256   | 0.256  |
| lr schedule | cosine  | cosine  |

#### Optimizer

| oneflow  | nvidia   |
| -------- | -------- |
| momentum | momentum |

#### Weight Initializer

OneFlow和英伟达保持了相同的初始化方式，只是在两个框架中部分api的名称不同。

| variable    | oneflow       | nvidia                       |
| ----------- | ------------- | ---------------------------- |
| conv weight | random_normal | Xavier( 'gaussian', 'in', 2) |
| conv bias   | NA            | NA                           |
| fc weight   | random_normal | Xavier( 'gaussian', 'in', 2) |
| fc bias     | 0             | 0                            |
| bn gamma    | 1             | 1                            |
| bn beta     | 0             | 0                            |

#### Weight Decay

| item         | oneflow   | nvidia    |
| ------------ | --------- | --------- |
| weight_decay | 1.0/32768 | 1.0/32768 |
| conv weight  | Yes       | Yes       |
| conv bias    | NA        | NA        |
| fc weight    | Yes       | Yes       |
| fc bias      | Yes       | NA        |
| bn gamma     | No        | No        |
| bn beta      | No        | No        |

#### Batch Norm

| param    | oneflow | nvidia |
| -------- | ------- | ------ |
| momentum | 0.9     | 0.9    |
| epsilon  | 1e-5    | 1e-5   |

#### Label Smoothing

| item            | oneflow | nvidia |
| --------------- | ------- | ------ |
| label smoothing | 0.1     | 0.1    |

###  数据集制作

#### 用于图像分类数据集简介

用于图像分类的公开数据集有CIFAR，ImageNet等等，这些数据集中，是以jpeg的格式提供原始的图片。

- [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)
  是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。包括CIFAR-10和CIFAR-100。

- [ImageNet](http://image-net.org/index) 
	ImageNet数据集，一般是指2010-2017年间大规模视觉识别竞赛(ILSVRC)的所使用的数据集的统称。ImageNet数据从2010年来稍有变化，常用ImageNet-2012数据集包含1000个类别，其中训练集包含1,281,167张图片，每个类别数据732至1300张不等，验证集包含50,000张图片，平均每个类别50张图片。



#### OFRecord提高IO效率

**原始的数据集：**

往往是由成千上万的图片或文本等文件组成，这些文件被散列存储在不同的文件夹中，一个个读取的时候会非常慢，并且占用大量内存空间。

**OFRecord：**

内部借助“Protocol Buffer”二进制数据编码方案，它只占用一个内存块，只需要一次性加载一个二进制文件的方式即可，简单，快速，尤其对大型训练数据很友好。另外，当我们的训练数据量比较大的时候，可以将数据分成多个OFRecord文件，来提高处理效率。

关于OFRecord的详细说明请参考：[OFRecord数据格式](https://github.com/Oneflow-Inc/oneflow-documentation/docs/basics_topics/ofrecord.md)



#### 将ImageNet转换成OFRecord

在OneFlow中，提供了将原始ImageNet-2012数据集文件转换成OFRecord格式的脚本。如果您已经准备好了ImageNet-2012数据集(训练集和验证集)，并且训练集/验证集的格式如下：

```shell
│   ├── train
│   │   ├── n01440764
│   │   └── n01443537
                                 ...
│   └── validation
│       ├── n01440764
│       └── n01443537
                                 ...
```

那么，一键执行以下脚本即可完成训练集和验证集 > OFRecord的转换：
##### 转换训练集

```shell
python3 imagenet_ofrecord.py  \
--train_directory ../data/imagenet/train  \
--output_directory ../data/imagenet/ofrecord/train   \
--label_file imagenet_lsvrc_2015_synsets.txt   \
--shards 256  --num_threads 8 --name train  \
--bounding_box_file imagenet_2012_bounding_boxes.csv   \
--height 224 --width 224
```

##### 转换验证集

```shell
python3 imagenet_ofrecord.py  \
--validation_directory ../data/imagenet/validation  \
--output_directory ../data/imagenet/ofrecord/validation  \
--label_file imagenet_lsvrc_2015_synsets.txt --name validation  \
--shards 256 --num_threads 8 --name validation \
--bounding_box_file imagenet_2012_bounding_boxes.csv  \
--height 224 --width 224
```

##### 参数说明

```shell
--train_directory
# 指定待转换的训练集文件夹路径
--validation_directory
# 指定待转换的验证集文件夹路径
--name
# 指定转换的是训练集还是验证集
--output_directory
# 指定转换后的ofrecord存储位置
 --num_threads
# 任务运行线程数
--shards
# 指定ofrecord分片数量，建议shards = 256
#（shards数量越大，则转换后的每个ofrecord分片数据量就越少）
--bounding_box_file
# 该参数指定的csv文件中标记了所有目标box的坐标，使转换后的ofrecord同时支持分类和目标检测任务
```

运行以上脚本后，你可以在../data/imagenet/ofrecord/validation、../data/imagenet/ofrecord/train下看到转换好的ofrecord文件：

```shell
.
├── train
│   ├── part-00000
│   └── part-00001
                             ...
└── validation
    ├── part-00000
    └── part-00001
                             ...
```



如果您尚未下载过Imagenet数据集，请自行下载和准备以下文件：

- ILSVRC2012_img_train.tar

- ILSVRC2012_img_val.tar

我们将用以下两个步骤，帮您完成数据集的预处理。之后，您就可以使用上面介绍的转换脚本进行OFReciord的转换了。下面假设您已经下载好了原始数据集，并存放在data/imagenet目录下：

```shell
├── data
│   └── imagenet
│       ├── ILSVRC2012_img_train.tar
│       ├── ILSVRC2012_img_val.tar
├── imagenet_utils
│   ├── extract_trainval.sh
│   ├── imagenet_2012_bounding_boxes.csv
│   ├── imagenet_2012_validation_synset_labels.txt
│   ├── imagenet_lsvrc_2015_synsets.txt
│   ├── imagenet_metadata.txt
│   ├── imagenet_ofrecord.py
│   └── preprocess_imagenet_validation_data.py
```

**步骤一：extract imagenet**

这一步主要是将ILSVRC2012_img_train.tar和ILSVRC2012_img_val.tar解压缩，生成train、validation文件夹。train文件夹下是1000个虚拟lebel分类文件夹(如：n01443537)，训练集图片解压后根据分类放入这些label文件夹中；validation文件夹下是解压后的原图。

```shell
sh extract_trainval.sh ../data/imagenet # 参数指定存放imagenet元素数据的文件夹路径
```
```shell
解压后，文件夹结构示意如下：
.
├── extract_trainval.sh
├── imagenet
│   ├── ILSVRC2012_img_train.tar
│   ├── ILSVRC2012_img_val.tar
│   ├── train
│   │   ├── n01440764
│   │   │   ├── n01440764_10026.JPEG
│   │   │   ├── n01440764_10027.JPEG 
                                               ...
│   │   └── n01443537
│   │       ├── n01443537_10007.JPEG
│   │       ├── n01443537_10014.JPEG
											 ...
│   └── validation
│       ├── ILSVRC2012_val_00000236.JPEG
│       ├── ILSVRC2012_val_00000262.JPEG        
											...
```

**步骤二：validation数据处理**

经过上一步，train数据集已经放入了1000个分类label文件夹中形成了规整的格式，而验证集部分的图片还全部堆放在validation文件夹中，这一步，我们就用preprocess_imagenet_validation_data.py对其进行处理，使其也按类别存放到label文件夹下。
```shell
python3 preprocess_imagenet_validation_data.py  ../data/imagenet/validation
# 参数 ../data/imagenet/validation为ILSVRC2012_img_val.tar解压后验证集图像存放的路径。
```
处理后项目文件夹格式如下：
```shell
.
├── extract_trainval.sh
├── imagenet
│   ├── ILSVRC2012_img_train.tar
│   ├── ILSVRC2012_img_val.tar
│   ├── train
│   │   ├── n01440764
│   │   └── n01443537
                                ...
│   └── validation
│       ├── n01440764
│       └── n01443537
                               ...
```

至此，已经完成了全部的数据预处理，您可以直接跳转至**转换训练集**和**转换验证集**部分，轻松完成ImageNet-2012数据集到OFRecord的转换过程了。


