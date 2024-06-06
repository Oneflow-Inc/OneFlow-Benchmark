#!/bin/bash
set -ex
if [ $# -ne 1 ]; then
  echo "Usage: $0 num_nodes"
  exit 1
fi
NUM_NODES="$1"
docker_name="cd_test_new"
ansible hosts -i inventory.ini -m shell -a "docker exec $docker_name bash -c 'cd /workspace/models/Vision/classification/image/resnet50 && bash train.sh $NUM_NODES'"