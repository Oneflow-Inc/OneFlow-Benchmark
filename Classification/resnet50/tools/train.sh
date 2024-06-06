NUM_NODES=${1:-1}

if [ "$NODE_RANK" -lt "$NUM_NODES" ]; then
  bash examples/args_train_ddp_graph_resnet50.sh "$NUM_NODES" 8 "$NODE_RANK" 192.168.1.27 /data/dataset/ImageNet/ofrecord 128 50 true python3 graph gpu 100 false '' 1
else
  echo do nothing
fi