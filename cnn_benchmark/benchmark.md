## Inference

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

### CV
batch size为8，输入图片大小为224，预热5 batches，平均吞吐（img/s）为500个batches的平均值。

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT (fp32) | TensorRT (fp16) | TensorRT (int8) |
>| ------------ | ------------- | ------------- | --------------- | --------------- | --------------- |
>| alexnet      | 2637          | 1550          | 2540            | 2759            |                 |
>| vgg16        | 371           | 332           | 377             | 1124            |                 |
>| resnet50     | 657           | 541           | 729             | 940             |                 |
>| inception-v3 |            |            |              |              |                 |
