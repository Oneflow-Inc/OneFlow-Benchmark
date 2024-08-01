#!/bin/bash

if [ -n "$1" ]; then
  docker_image_path=$1
else
  docker_image_path="/share_nfs/k85/oneflow.0.9.1.dev20240203-cuda11.8.tar"
fi

if [ -n "$2" ]; then
  docker_image_tag=$2
else
  docker_image_tag="oneflowinc/oneflow:0.9.1.dev20240203-cuda11.8"
fi

if [ -n "$3" ]; then
  force_load=$3
else
  force_load=false
fi

ansible-playbook \
    -i ../inventory.ini \
    load_and_tag_docker_image.yml \
    -e "docker_image_path=$docker_image_path" \
    -e "docker_image_tag=$docker_image_tag" \
    -e "force_load=$force_load"
