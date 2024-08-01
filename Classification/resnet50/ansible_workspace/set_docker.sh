#!/bin/bash
set -ex
if [ $# -ne 1 ]; then
  echo "Usage: $0 filename"
  exit 1
fi
host_file="$1"
num_hosts=$(wc -l < "$host_file")
docker_name="cd_test_new"

mapfile -t lines < "$host_file"

for (( i=1; i<${#lines[@]}; i++ )); do
  line="${lines[$i]}"
  host_name=$(echo "$line" | awk '{print $1}')
  ansible $host_name -i $host_file -m shell -a "docker run -itd -e NODE_RANK=$((i-1)) -v /data/dataset/ImageNet:/data/dataset/ImageNet -v /data/home/chende/tools:/workspace/tools --network host --gpus all --shm-size=16g --ulimit memlock=-1 --ulimit core=0 --ulimit stack=67108864 --privileged --ipc host --cap-add=IPC_LOCK --name $docker_name nvcr.io/nvidia/pytorch:24.03-py3 bash"
done
ansible hosts -i "$host_file" -m shell -a "docker exec $docker_name bash -c 'bash /workspace/tools/prepare_docker.sh'"