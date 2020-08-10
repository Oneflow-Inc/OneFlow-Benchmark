# OneFlow Deep Learning Benchmarks
## Introduction
This repository provides OneFlow deep learning benchmark examples for CV, CTR and NLP, and more models are on the way and will be provided here when ready.

## [Convolutional Networks](./Classification/cnns) for Computer Vision Classification
- [ResNet-50](./Classification/cnns)
- [ResNeXt-50-32*4d](./Classification/cnns)
- [VGG-16](./Classification/cnns)
- [Inception-V3](./Classification/cnns)
- [AlexNet](./Classification/cnns)
- [MobileNet-V2](./Classification/cnns)

## [Wide Deep Learning](./ClickThroughRate/WideDeepLearning) for Click-Through-Rate (CTR) Recommender Systems
- [OneFlow-WDL](./ClickThroughRate/WideDeepLearning)

## [BERT](./LanguageModeling/BERT) for Nature Language Process
- [BERT Pretrain for Language Modeling](./LanguageModeling/BERT)
- [SQuAD for Question Answering](./LanguageModeling/BERT)
- [CoLA and MRPC of GLUE](./LanguageModeling/BERT)

## OneFlow Benchmark Test Reports

| Model | DType | XLA | Throughput | Speedup on 32 devices |
| ----- | ----- | --- | ---------- | ------- |
| [ResNet50-V1.5](./reports/resnet50_v15_fp32_report.md) | Float32 | No | 11.6k imges/sec | 30.4 |
| [BERT base Pretrain](./reports/bert_fp32_report.md) | Float32 | No | 530k tokens/sec | 28.54 |
