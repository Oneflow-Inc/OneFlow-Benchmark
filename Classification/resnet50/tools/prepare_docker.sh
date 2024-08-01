#!/bin/bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install --upgrade pip
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu121

cd /workspace 
cp tools/models.tar.gz ./
tar -xvf models.tar.gz
pip install -r models/dev-requirements.txt
pip install -r models/Vision/classification/image/resnet50/requirements.txt

cp tools/args_train_ddp_graph_resnet50.sh models/Vision/classification/image/resnet50/examples/
cp tools/train.sh models/Vision/classification/image/resnet50/
cp tools/profile.sh models/Vision/classification/image/resnet50/