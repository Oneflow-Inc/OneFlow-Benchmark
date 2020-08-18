# OneFlow 深度学习基准

 [![](https://img.shields.io/badge/Language-EN-blue.svg)](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/dev_sx_benchmark)

本仓库将提供一系列由 OneFlow 最新的高级接口实现的模型及样例网络，模型基准涉及计算机视觉（Computer Vision，CV）、点击率推荐（Click-Through-Rate，CTR）、自然语言处理（Natural Language Processing， NLP）。

为了能让 OneFlow 用户实现自己拓展研究和产品迭代的需求，充分利用该框架，本仓库旨在提供各个模型基于 OneFlow 的最佳实现。

更多新模型正在路上！

## 内容

### 模型及实现

- ### 计算机视觉（Computer Vision）

  - #### 图片识别（Image Classification）

| 模型                                                         | 参考来源（论文）                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ResNet-50](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/resnet_model.py) | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| [ResNeXt](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/resnext_model.py) | [Aggregated_Residual_Transformations_CVPR_2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) |
| [VGG-16](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/vgg_model.py) | [VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/) |
| [Inception-V3](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/inception_model.py) | [Inception V3 Deep Convolutional Architecture For Classifying Acute Myeloid/Lymphoblastic Leukemia](https://software.intel.com/content/www/us/en/develop/articles/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic.html) |
| [AlexNet](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/alexnet_model.py) | [ImageNet Classification with Deep Convolutional Neural Networks](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf) |
| [MobileNet-V2](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/mobilenet_v2_model.py) | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) |

- ### 自然语言处理（Natural Language Processing）

| 模型                                                         | 参考来源（论文）                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BERT (Bidirectional Encoder Representations from Transformers)](https://github.com/OneFlow/models/blob/master/official/nlp/bert) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| [SQuAD for Question Answering](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/BERT/run_squad.py) | [BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD)        |
| [CoLA and MRPC of GLUE](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/BERT/run_classifier.py) | [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://www.aclweb.org/anthology/W18-5446.pdf) |

- ### 点击率推荐（Click-Through-Rate）

  | 模型                                                         | 参考来源（论文）                                             |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | [OneFlow-WDL](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning) | [**Wide & Deep Learning for Recommender Systems**](https://arxiv.org/pdf/1606.07792.pdf) |



### 开始使用模型

...
