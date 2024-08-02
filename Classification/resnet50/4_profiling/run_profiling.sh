#!/bin/bash

DOCKER_IMAGE="oneflowinc/oneflow:0.9.1.dev20240203-cuda11.8"
SRC="/share_nfs/k85/models/Vision/classification/image/resnet50"

if [ -n "$1" ]; then
  DOCKER_IMAGE="$1"
fi

if [ -n "$2" ]; then
  SRC="$2"
fi

# 运行 ansible-playbook 命令
ansible-playbook -i ../inventory.ini profiling.yml -e "docker_image=${DOCKER_IMAGE}" -e "src=${SRC}"
