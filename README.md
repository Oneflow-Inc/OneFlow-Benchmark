# OneFlow Deep Learning Benchmarks

 [![](https://img.shields.io/badge/Language-CH-red.svg)](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/dev_sx_benchmark/README_CH.md)

This repository provides a collection of example implementations and modeling solutions using the latest OneFlow's high-level APIs, for CV, CTR and NLP models as a benchmark.

It aims to demonstrate the best practices for modeling so that OneFlow users can take full advantage of OneFlow for their research and product development.

 More models are coming!

## Contents

### Models and Implementations

- ### Computer Vision

  - #### Image Classification

| Model                                                        | Reference (Paper)                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ResNet-50](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/resnet_model.py) | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| [ResNeXt](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/resnext_model.py) | [Aggregated_Residual_Transformations_CVPR_2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) |
| [VGG-16](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/vgg_model.py) | [VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/) |
| [Inception-V3](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/inception_model.py) | [Inception V3 Deep Convolutional Architecture For Classifying Acute Myeloid/Lymphoblastic Leukemia](https://software.intel.com/content/www/us/en/develop/articles/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic.html) |
| [AlexNet](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/alexnet_model.py) | [ImageNet Classification with Deep Convolutional Neural Networks](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf) |
| [MobileNet-V2](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/mobilenet_v2_model.py) | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) |

- ## Natural Language Processing

| Model                                                        | Reference (Paper)                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [BERT (Bidirectional Encoder Representations from Transformers)](https://github.com/OneFlow/models/blob/master/official/nlp/bert) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| [SQuAD for Question Answering](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/BERT/run_squad.py) | [BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD)        |
| [CoLA and MRPC of GLUE](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/LanguageModeling/BERT/run_classifier.py) | [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://www.aclweb.org/anthology/W18-5446.pdf) |

- ## Click-Through-Rate 

| Model                                                        | Reference (Paper)                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [OneFlow-WDL](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning) | [**Wide & Deep Learning for Recommender Systems**](https://arxiv.org/pdf/1606.07792.pdf) |



## Get started with the models

- The models in the master branch are developed using OneFlow [], and they target the OneFlow [nightly binaries](https://github.com/OneFlow/OneFlow#installation) built from the [master branch of OneFlow](https://github.com/OneFlow/OneFlow/tree/master).

- The stable versions targeting releases of OneFlow are available as tagged branches or [downloadable releases](https://github.com/OneFlow/models/releases).

  

Please follow the below steps before running models in this repository.

### Requirements

- Python >= 3.5

- CUDA Toolkit Linux x86_64 Driver

  | OneFlow       | CUDA Driver Version |
  | ------------- | ------------------- |
  | oneflow_cu102 | >= 440.33           |
  | oneflow_cu101 | >= 418.39           |
  | oneflow_cu100 | >= 410.48           |
  | oneflow_cu92  | >= 396.26           |
  | oneflow_cu91  | >= 390.46           |
  | oneflow_cu90  | >= 384.81           |

  - CUDA runtime is statically linked into OneFlow. OneFlow will work on a minimum supported driver, and any driver beyond. For more information, please refer to [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).
  - Support for latest stable version of CUDA will be prioritized. Please upgrade your Nvidia driver to version 440.33 or above and install `oneflow_cu102` if possible.
  - We are sorry that due to limits on bandwidth and other resources, we could only guarantee the efficiency and stability of `oneflow_cu102`. We will improve it ASAP.

### Installation

#### Method 1: Install with pip package

- To install latest release of OneFlow with CUDA support:

  ```
  python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu102 --user
  ```

- To install OneFlow with legacy CUDA support, run one of:

  ```
  python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu101 --user
  python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu100 --user
  python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu92 --user
  python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu91 --user
  python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu90 --user
  ```

- If you are in China, you could run this to have pip download packages from domestic mirror of pypi:

  ```
  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

  For more information on this, please refer to [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)

- CPU-only OneFlow is not available for now.

- Releases are built with G++/GCC 4.8.5, cuDNN 7 and MKL 2020.0-088.

#### Method 2: Build from source

1. System Requirements to Build OneFlow

- Please use a newer version of CMake to build OneFlow. You could download cmake release from [here](https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz).

- Please make sure you have G++ and GCC >= 4.8.5 installed. Clang is not supported for now.

- To install dependencies, run:

  ```
  yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo && \
  rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
  yum update -y && yum install -y epel-release && \
  yum install -y intel-mkl-64bit-2020.0-088 nasm swig rdma-core-devel
  ```

  On CentOS, if you have MKL installed, please update the environment variable:

  ```
  export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
  ```

  If you don't want to build OneFlow with MKL, you could install OpenBLAS:

  ```
  sudo yum -y install openblas-devel
  ```

2. Clone Source Code

Clone source code and submodules (faster, recommended)

```
git clone https://github.com/Oneflow-Inc/oneflow
cd oneflow
git submodule update --init --recursive
```

Or you could also clone the repo with `--recursive` flag to clone third_party submodules together

```
git clone https://github.com/Oneflow-Inc/oneflow --recursive
```

3. Build and Install OneFlow

```
cd build
cmake ..
make -j$(nproc)
make pip_install
```

- For pure CPU build, please add this CMake flag `-DBUILD_CUDA=OFF`.

### More models to come

[new models]

## Contributions

- How to add new models?
- How to add new framework tests?

| Model | DType | XLA | Throughput | Speedup on 32 devices |
| ----- | ----- | --- | ---------- | ------- |
| [ResNet50-V1.5](./reports/resnet50_v15_fp32_report.md) | Float32 | No | 11.6k imges/sec | 30.4 |
| [BERT base Pretrain](./reports/bert_fp32_report.md) | Float32 | No | 530k tokens/sec | 28.54 |
