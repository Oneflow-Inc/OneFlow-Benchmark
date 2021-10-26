## Directory description
```
.
├── gpu_info                Inside are temporary csv files that include gpu usage situation
├── log
├── n1g8_benchmark.sh       test loss in multi-gpus
├── n1g1_benchmark.sh       test loss in one gpu
├── n1g8_benchmark_mem.sh   test memory and latency in multi-gpus
├── README.md
└── wdl_train_eval.py       python script
```


## How to start a test task
We use n1g1_benchmark as an example:

`./n1g1_benchmark.sh`, then a csv file named *n1g1_benchmark* will be created under the */results/old* directory.
