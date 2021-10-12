## Directory description
```
.
├── eval
│   ├── csv
│   │   ├── gpu_info        Inside are temporary csv files that include gpu usage situation
│   │   ├── n1g1_ddp        Every test task result, multiple csv files and the number is gpu_num
│   ├── eval.py             python script
│   ├── main.py             python script
│   ├── n1g1_ddp_mem.sh     test memory and latency in one gpu
│   ├── n1g1_ddp.sh         test loss in one gpu
│   ├── n1g1_eager_mem.sh   test memory and latency in one gpu
│   ├── n1g1_eager.sh       test loss in one gpu
│   ├── n1g1_graph_mem.sh   test memory and latency in one gpu
│   ├── n1g1_graph.sh       test loss in one gpu
│   ├── n1g8_ddp_mem.sh     test memory and latency in multi-gpus
│   ├── n1g8_ddp.sh         test loss in multi-gpus
│   ├── n1g8_eager_mem.sh   test memory and latency in multi-gpus
│   ├── n1g8_eager.sh       test loss in multi-gpus
│   ├── n1g8_graph_mem.sh   test memory and latency in multi-gpus
│   ├── n1g8_graph.sh       test loss in multi-gpus
├── config.py
├── graph.py
├── models
│   ├── dataloader_utils.py
│   └── wide_and_deep.py
├── README.md
└── util.py                 merge param from old version
```

## How to start a test task
We use n1g1_ddp as an example:

`./n1g1_ddp.sh`, then a directory named *n1g1_ddp* which contains multi csv files will be created under the *csv* directory, and each csv file corresponds to training info in one device.

Remember to copy the directory to results/new if you want to analyze them further.