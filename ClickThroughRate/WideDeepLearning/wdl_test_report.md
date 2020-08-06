[HugeCTR](https://github.com/NVIDIA/HugeCTR) is a recommender specific framework provided by NVIDIA Corporation. It is designed for Click-Through-Rate (CTR) estimation.

OneFlow build up Wide & Deep Learning (WDL) network based on HugeCTR. 

OneFlow-WDL network supports model parallelism and sparse gradient update. It can support over 400 million vocab size of lookup table in a TitanV 12G * 8 server, at the same time has the same performace with small vocab size table.

The purpose of this document is to introduce how to use OneFlow-WDL to train network and present the testing results of OneFlow-WDL.  

## Environment and Preparation 
Please make sure to install OneFlow in your computer/server before running OneFlow-WDL, and [scikit-learn](https://scikit-learn.org/stable/install.html) is required to calculate metrics.

### Requirements
- python 3.x（recommended）
- OneFlow 
- scikit-learn

### Data preparation 
A small [data set](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/criteo_wdl_3000w_ofrecord_example.tgz) is prepared for your fast evaluation. Following is the folder structure of this example dataset. 
```
criteo_wdl_3000w_ofrecord_example
├── train
│   └── part-00000
└── val
    ├── part-00000
    └── README.md
```

Making a full-size dataset is exhausting. Hopefully [*Use Spark to create WDL dataset*](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/how_to_make_ofrecord_for_wdl.md) can help you to generate the full-size ofrecord for testing. You can download original dataset from [CriteoLabs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and follow the steps in [Spark 2.4.* shell](https://www.apache.org/dyn/closer.lua/spark/spark-2.4.6/spark-2.4.6-bin-hadoop2.7.tgz).

### OneFlow-WDL code
The main code we test is the file: `wdl_train_eval.py`. Please download from [here](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/wdl_train_eval.py).

## Run OneFlow-WDL code
```
VOCAB_SIZE=1603616
DATA_ROOT=/path/to/wdl/criteo_wdl_3000w_ofrecord_example
python3 wdl_train_eval.py \
  --train_data_dir $DATA_ROOT/train \
  --train_data_part_num 1 \
  --train_part_name_suffix_length=5 \
  --eval_data_dir $DATA_ROOT/val \
  --eval_data_part_num 1 \
  --max_iter=300000 \
  --loss_print_every_n_iter=1000 \
  --eval_interval=1000 \
  --batch_size=16384 \
  --wide_vocab_size=$VOCAB_SIZE \
  --deep_vocab_size=$VOCAB_SIZE \
  --gpu_num 1
```

The running shell code is shown above, the only thing we need to config is `DATA_ROOT` of ofrecord dataset for OneFlow-WDL. Then the shell is able to run. If the following output show up which means the code is running correctly.

Note: the `criteo_wdl_3000w_ofrecord_example` dataset has one part file only, the `train_data_part_num` and `eval_data_part_num` are both set to `1`.
```
1000 time 2020-07-08 00:28:08.066281 loss 0.503295350909233
1000 eval_loss 0.4846755236387253 eval_auc 0.7616240146992771
2000 time 2020-07-08 00:28:11.613961 loss 0.48661992555856703
2000 eval_loss 0.4816856697201729 eval_auc 0.765256583562705
3000 time 2020-07-08 00:28:15.149135 loss 0.48245503094792364
3000 eval_loss 0.47835959643125536 eval_auc 0.7715609382514008
4000 time 2020-07-08 00:28:18.686327 loss 0.47975033831596375
4000 eval_loss 0.47925308644771575 eval_auc 0.7781267916810946
```
## Testing results and explanation
All tests are performed on a sever with 8 TitanV 12G GPUs installed. As a reference, we perform some Nvidia HugeCTR tests in docker container.

### Multi-devices performance
The main purpose of this test is to test the average latency over different GPU device number with fixed total batch size = 16384. 7 hidden layers with 1024 neural units are applied in this test.

Results：

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/fixed_batch_size_latency.png)

the maximum memory usage over devices is shown below:

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/fixed_batch_size_memory.png)

To summarise, from one device to 8 devices, OneFlow-WDL ran faster than HugeCTR with less memory usage.

### Batch size per device = 16384 , multi-devices performance
The main purpose of test is to test the average latency over different GPU devices with batch size=16384 per device, the total batch size is scaled with device number. 7 hidden layers with 1024 neural units are applied in this test.

Results：

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/scaled_batch_size_latency.png)

the maximum memory usage over devices is shown below:

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/scaled_batch_size_memory.png)

Summary:
- The latency kept increase alone with number of devices.
- OneFlow-WDL ran faster than HugeCTR with less less memory consumption.
- There is no obvious change in memory usage.

### Performance in different batch size with one GPU device
The main purpose of this test is to test the average latency with one GPU device over different batch size. 2 hidden layers with 1024 neural units are applied in this test. 

Results：

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/scaled_batch_size_latency_1gpu.png)

Summary: OneFlow-WDL ran faster than HugeCTR over batch size from 512 to 16384.

### Big vocab size performance  
There are two Embedding Tables config in OneFlow-WDL：
- The size of `wide_embedding` is vocab_size x 1
- The size of`deep_embedding` is vocab_size x 16

In HugeCTR the vocab size is 1,603,616(1.6 million). We kept increasing vocab size from 3.2 million to 409.6 million during test, result is below：

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/big_vocab_table_2x1024.png) 

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/big_vocab_table_7x1024.png)

In above figures，the blue column is average latency and orange curve is for the memory usage over different vocab size.

Conclusion: with the increaseing of vocab size, memory usage increase, but the average latency kept still.

Our test GPU has 12G memory only, we can image how big vocab size will OneFlow-WDL support with 16G, 32G or even larger memory devices. **409.6 Million vocab size is not the limitation but a begining**. 

### Convergence test 1
We choose batch size=512 to run the convergence performance test. 

The follow graph is the results of first 500 iterations. We perform evaluation with 20 example after each iteration.

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/eval_auc_loss_500iters.png)

Conclusion: AUC grow rapidly over 0.75.

### Convergence test 2
Same with the Convergence test 1, but we print the average loss value every 1000 iterations, then select 20 record to evaluate. 300,000 training iterations in total. Result:

![image](https://github.com/Oneflow-Inc/oneflow-documentation/raw/master/cn/docs/adv_examples/imgs/train_eval_auc_loss.png)

Conclusion and analysis:
1. The blue curve of train loss have obvious descend. because, there are 36674623 data in training set. When batch_size=512, 71630 steps will finish a epoch. 300,0000 steps can use the training set over 4 times(epochs). The descend of blue curve proof that. OneFlow can suffle the data during the training process in order to reduce overfitting. 
2. The orange curve is evaluation loss. It maintains descend in first two epochs and began to ascend in third epoch because of overfitting. 
3. The grey curve is the AUC of evaluation set. AUC also meet the peak in second epoch which over 0.8. Then descend in next few epoch.
