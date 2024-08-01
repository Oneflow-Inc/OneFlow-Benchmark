#!/bin/bash

if [ -n "$1" ]; then
  ansible-playbook -i ../inventory.ini pull_docker_image.yml -e "docker_image=$1"
else
  ansible-playbook -i ../inventory.ini pull_docker_image.yml
fi
