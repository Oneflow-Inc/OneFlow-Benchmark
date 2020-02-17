[TOC]

# Inference

测试平台：Nvidia GTX2080Ti单卡.  
CUDA版本：10.0  
CUDNN版本：7.5.0   
TensorRT版本：6.0.1  

Oneflow-Benchmark   
branch: of_dev_python_py3    
commit: 985dd3f03887d266e66573db0b31a4cf3051ff31   

Oneflow:   
branch: of_xrt_tensorrt   
commit: 726c3a12b9d97b57f9fb7e3d212b63564e20e755   

## CV

### Speed

输入图片大小为224 (inception-v3为299)，预热5 batches，平均吞吐（img/s）为500个batches的平均值。

#### 1. batch size为8

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) | TensorRT official(fp32) | TensorRT official(fp16) | TensorRT official(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- | ----------------------- | ----------------------- | ----------------------- |
>| alexnet      | 2637          | 1550          | 2540           | 2759           |                |                         |                         |                         |
>| vgg16        | 371           | 332           | 377            | 1124           |                | 470                     | 1629                    |                         |
>| resnet50     | 657           | 541           | 729            | 940            |                | 1025                    | 2500                    |                         |
>| inception-v3 | 433           | 434           | 489            | 999            |                |                         |                         |                         |

- 对vgg16 fp32结果的分析：
  1. oneflow tensorrt的吞吐为377 images/s，即21.2 ms/batch，而tensorrt官方测的吞吐为470 images/s，即17.02 ms/batch。
  2. 去除图片解码的耗时，oneflow tensorrt的吞吐可以达到407 images/s，即19.65 ms/batch。
  3. 根据profile的结果，oneflow tensorrt每隔两个batch会出现2-3ms的空隙，平均每个batch 1-1.5 ms。
  4. tensorrt不会自动做conv和bias的融合，需要手动将bias合并到conv中，tensorrt官方的benchmark是conv和bias融合后的结果。这里大概能优化2 ms左右。
  5. 总结：对2、3和4进行优化后，理论上一个batch的耗时能达到 (19.65 - 1 - 2 =) 16.65 ms，与tensorrt官方测试的结果基本一致。

- 对resnet50 fp32结果的分析：
  1. 同样去除图片解码的耗时，oneflow tensorrt的吞吐可以达到800 images/s, 即10 ms/batch, 而tensorrt官方测的吞吐为1025 images/s，即7.8 ms/batch。
  2. oneflow tensorrt没有支持batch normalization，导致整图被分割成多个tensorrt子图。如果oneflow tensorrt支持batch normalization，将减少4ms左右。但同时发现支持了batch normalization后，batch之间的空隙从几乎0ms增加到了5.32ms，导致即使支持了batch normalization后，吞吐并没有明显的变化。
  3. 总结：如果对1、2优化后，理论上一个batch的耗时能达到 (10 - 4 =) 6ms。

- Update 2019.12.24: 所有source op都通过device tick代理到cpu tick，减少event次数。
- Update 2020.2.17: 增加int8 benchmark。

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) | TensorRT official(fp32) | TensorRT official(fp16) | TensorRT official(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- | ----------------------- | ----------------------- | ----------------------- |
>| alexnet      | 2692          | 2022          | 2679           | 4060           | 5896           |                         |                         |                         |
>| vgg16        | 398           | 346           | 425            | 1200           | 2054           | 470                     | 1629                    |                         |
>| resnet50     | 735           | 570           | 945            | 2120           | 3512           | 1025                    | 2500                    |                         |
>| inception-v3 | 538           | 510           | 572            | 1356           | 2094           |                         |                         |                         |


#### 2. batch size为50

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) | TensorRT official(fp32) | TensorRT official(fp16) | TensorRT official(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- | ----------------------- | ----------------------- | ----------------------- |
>| alexnet      | 6999          | 3219          | 4306           | 7704           |                |                         |                         |                         |
>| vgg16        | 497           | 476           | 404            | 1482           |                | 498                     | 1907                    |                         |
>| resnet50     | 810           | 619           | 830            | 1285           |                | 1302                    | 3843                    |                         |
>| inception-v3 | 544           | 531           | 717            | 1839           |                |                         |                         |                         |

- Update 2019.12.24：所有source op都通过device tick代理到cpu tick，减少event次数。
- Update 2020.2.17: 增加int8 benchmark。

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) | TensorRT official(fp32) | TensorRT official(fp16) | TensorRT official(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- | ----------------------- | ----------------------- | ----------------------- |
>| alexnet      | 6568          | 3341          | 5030           | 9076           | 14378          |                         |                         |                         |
>| vgg16        | 528           | 498           | 459            | 1638           | 2817           | 498                     | 1907                    |                         |
>| resnet50     | 888           | 685           | 1262           | 3989           | 8239           | 1302                    | 3843                    |                         |
>| inception-v3 | 698           | 589           | 797            | 2363           | 4022           |                         |                         |                         |

### Precision

总共5w张图片, 统计Top1 accuracy和相对oneflow fp32的分类误差数量。

- Update 2020.2.17: 增加int8 benchmark。

>|  -           | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- |
>| vgg16        | 0.495 / 0     | 0.495 / 61    | 0.495 / 0      | 0.495 / 101    | 0.493          |
>| alexnet      |               |               |                |                |                |
>| resnet50     | 0.613 / 0     | 0.613 / 59    | 0.613 / 0      | 0.613 / 130    | 0.614          |
>| inception-v3 |               |               |                |                |                |

