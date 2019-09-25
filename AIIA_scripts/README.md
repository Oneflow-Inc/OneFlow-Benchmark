# Benchmark Scripts for AIIA Testing
## use cases
* single node:
    ```
    python3 run_benchmark.py --model=vgg16,resnet50,bert --case=1n1c,1n2c
    ```

* multi nodes:

    node1(192.168.1.12):
    ```
    python3 run_benchmark.py --model=vgg16,resnet50,bert --case=2n4c --node_list=192.168.1.12,192.168.1.14
    ```
    node2(192.168.1.14):
    ```
    python3 run_benchmark.py --model=vgg16,resnet50,bert --case=2n4c --node_list=192.168.1.12,192.168.1.14
    ```
    The result only print on node1.

* only run synthetic data for vgg16:
    ```
    python3 run_benchmark.py --model=vgg16 --case=1n1c --run_real_data=False --run_synthetic_data=True
    ```

## useful tools
* extract all benchmarking results,use:
    ```
    python3 tools/extract_result.py
    ```
