
## single node:
```
python run_benchmark.py --model=vgg16,resnet50,bert --case=1n1c,1n2c
```

## multi nodes:
node1(192.168.1.12):
```
python run_benchmark.py --model=vgg16,resnet50,bert --case=2n4c --node_list=192.168.1.12,192.168.1.14
```
node2(192.168.1.14):
```
python run_benchmark.py --model=vgg16,resnet50,bert --case=2n4c --node_list=192.168.1.12,192.168.1.14
```
The result only print on node1.

## only real data for vgg16:
```
python run_benchmark.py --model=vgg16 --case=1n1c --run_real_data=True --run_synthetic_data=false
```

## useful tools:
extract all benchmarking results:
```
python tools/extract_result.py
```