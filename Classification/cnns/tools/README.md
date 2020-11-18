# Tools使用说明
##  简介
tools文件夹中存放的文件和python代码专门用于 **ImageNet(2012)数据集** 制作工具。通过下面的使用说明，你可以将ImageNet(2012)从原始格式转换为通用图像格式的数据集，再转换为可在OneFlow中直接训练的 **OFRecord** 格式。

#### 原始数据集

往往是由成千上万的图片或文本等文件组成，这些文件被散列存储在不同的文件夹中，一个个读取的时候会非常慢，并且占用大量内存空间。

#### OFRecord
 **OFRecord提高IO效率** 

内部借助“Protocol Buffer”二进制数据编码方案，它只占用一个内存块，只需要一次性加载一个二进制文件的方式即可，简单，快速，尤其对大型训练数据很友好。另外，当我们的训练数据量比较大的时候，可以将数据分成多个OFRecord文件，来提高处理效率。

关于OFRecord的详细说明请参考：[OFRecord数据格式](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/cn/docs/extended_topics/ofrecord.md)



##  数据集制作

### 将ImageNet转换成OFRecord

在OneFlow中，提供了将原始ImageNet2012数据集文件转换成OFRecord格式的脚本，如果您已经下载过，且准备好了ImageNet2012通用图像格式的数据集，并且训练集/验证集的格式如下：

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

那么，您只需要下载：[imagenet_2012_bounding_boxes.csv](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/imagenet_2012_bounding_boxes.zip) 

然后执行以下脚本即可完成训练集/验证集 > OFRecord的转换：

#### 转换训练集

```shell
python3 imagenet_ofrecord.py  \
--train_directory ../data/imagenet/train  \
--output_directory ../data/imagenet/ofrecord/train   \
--label_file imagenet_lsvrc_2015_synsets.txt   \
--shards 256  --num_threads 8 --name train  \
--bounding_box_file imagenet_2012_bounding_boxes.csv   \
--height 224 --width 224
```

#### 转换验证集

```shell
python3 imagenet_ofrecord.py  \
--validation_directory ../data/imagenet/validation  \
--output_directory ../data/imagenet/ofrecord/validation  \
--label_file imagenet_lsvrc_2015_synsets.txt --name validation  \
--shards 256 --num_threads 8 --name validation \
--bounding_box_file imagenet_2012_bounding_boxes.csv  \
--height 224 --width 224
```

#### 参数说明

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



如果尚未下载/处理过ImageNet，请看下面【ImageNet的下载和预处理】部分的说明。

### ImageNet的下载和预处理

如果您尚未下载过Imagenet数据集，请准备以下文件：

- ILSVRC2012_img_train.tar
- ILSVRC2012_img_val.tar
- ILSVRC2012_bbox_train_v2.tar.gz（非必须）

其中训练集和验证集的图片请自行下载，bbox标注可以点此下载：[ILSVRC2012_bbox_train_v2.tar.gz](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/ILSVRC2012_bbox_train_v2.tar.gz)

我们将用下面三个步骤，帮您完成数据集的预处理。之后，您就可以使用【将ImageNet转换成OFRecord】部分介绍的转换脚本进行OFReciord的转换了。



下面假设您已经下载好了原始数据集和bbox标注文件，并存放在data/imagenet目录下：

```shell
├── data
│   └── imagenet
│       ├── ILSVRC2012_img_train.tar
│       ├── ILSVRC2012_img_val.tar
│       ├── ILSVRC2012_bbox_train_v2.tar.gz
├── tools
│   ├── extract_trainval.sh
│   ├── imagenet_2012_validation_synset_labels.txt
│   ├── imagenet_lsvrc_2015_synsets.txt
│   ├── imagenet_metadata.txt
│   ├── imagenet_ofrecord.py
│   └── preprocess_imagenet_validation_data.py
```

#### 步骤一：process_bounding_boxes

这一步，主要是将标注好的包含bboxs的xml文件提取到一个.csv文件中，方便后面代码中直接使用。完整的转换过程大约需要5分钟。

当然，你也可以直接使用我们转换好的文件：[imagenet_2012_bounding_boxes.csv](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/imagenet_2012_bounding_boxes.zip)

1.解压ILSVRC2012_bbox_train_v2.tar.gz

```shell
cd data/imagenet && mkdir bounding_boxes && tar -zxvf ILSVRC2012_bbox_train_v2.tar.gz -C bounding_boxes
```

2.提取bboxs至.csv文件

```shell
cd ../.. && python process_bounding_boxes.py  data/imagenet/bounding_boxes   imagenet_lsvrc_2015_synsets.txt  | sort > imagenet_2012_bounding_boxes.csv
```

#### 步骤二：extract imagenet

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
│   ├── ILSVRC2012_bbox_train_v2.tar.gz
│   ├── bounding_boxes
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

#### 步骤三：validation数据处理

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
│   ├── ILSVRC2012_bbox_train_v2.tar.gz
│   ├── bounding_boxes
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
