docker run -it --rm \
    -v /dataset/:/dataset/ \
    -v ../Oneflow-benchmark/:/workspace/benchmark/ \
    --network host \
    oneflow-benchmark 

