## Directory description
```
.
├── n1_benchmark.sh       test loss in one gpu
├── README.md
└── wdl_train_eval.py       python script
```

## How to start a test task
First, download [initial_model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/wdl_initial_model/checkpoint_old.zip) and unzip in this directory.

Then run `bash ./n1_benchmark.sh`, and a csv file named *n1_benchmark* will be created under the */log/* directory.
