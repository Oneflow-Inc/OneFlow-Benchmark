#!/bin/bash
echo "/data/home/chende/tools"
ansible hosts -i inventory.ini -m copy -a "src=/data/home/chende/tools dest=/data/home/chende/ mode=0755"