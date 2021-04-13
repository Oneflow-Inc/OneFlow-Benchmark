# Resnet50


## Predict on single image

```bash
bash predict.sh
```

## Train on [imagenette](https://github.com/fastai/imagenette) Dataset

### Prepare ofrecord

```bash
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/imagenette_ofrecord_224.tar.gz
tar zxf imagenette_ofrecord_224.tar.gz
```

### Run Training script

```bash
python3 train.py
```

