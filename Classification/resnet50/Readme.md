# 千卡 0.85

[toc]

## 文件目录结构
```
├── ansible_workspace # 主节点上的工作目录
│   ├── inventory.ini # 用来配置节点信息
│   ├── set_docker.sh # 在各节点上创建docker，并且配置好docker内环境
│   ├── profile.sh # 根据节点数启动profile
│   ├── train.sh # 根据节点数启动训练
│   └── update_tools.sh # 将主节点的tools文件夹复制到各个子节点
├── tools # 在各个节点使用的文件
│   ├── args_train_ddp_graph_resnet50.sh # 接受模型训练参数并启动训练
│   ├── models.tar.gz # 模型，为防止git网络问题，所以先下载放在共享目录下
│   ├── extract.py # 提取log中train阶段的throughput的平均值
│   ├── prepare_docker.sh # 用于配置docker内环境
│   ├── profile.sh # 根据节点数在本机启动profile
│   └── train.sh # 根据节点数在本机启动训练
└── Readme.md
```

需求：有NVLink，以及 shared_nfs

以下供参考

## 第一步： 配置环境

### 1.1 所有节点配置SSH Key，并设置authorized_keys

（怎么自动化）

需要一个共享的存储空间，如：`/shared_nfs/k85`，在一个文件夹下准备好 

- authorized_keys : 在主节点运行 

  ```bash
  #!/bin/bash
  
  # 设置 SSH 目录路径
  SSH_DIR="$HOME/.ssh"
  
  # 检查 SSH 目录是否存在，如果不存在则创建
  if [ ! -d "$SSH_DIR" ]; then
    mkdir -p "$SSH_DIR"
    echo "Created directory: $SSH_DIR"
  fi
  
  # 设置密钥文件路径
  KEY_PATH="$SSH_DIR/id_rsa"
  
  # 生成 SSH 密钥对
  ssh-keygen -t rsa -b 2048 -f "$KEY_PATH" -N "" -q
  
  # 创建 authorized_keys 文件
  cat $SSH_DIR/id_rsa.pub > $SSH_DIR/authorized_keys
  
  # 将 authorized_keys 文件拷贝到共享目录
  cp $SSH_DIR/authorized_keys shared_nfs/k85
  ```

- 在子节点运行

  ```bash
  #!/bin/bash
  
  # 设置 SSH 目录路径
  SSH_DIR="$HOME/.ssh"
  
  # 检查 SSH 目录是否存在，如果不存在则创建
  if [ ! -d "$SSH_DIR" ]; then
    mkdir -p "$SSH_DIR"
    echo "Created directory: $SSH_DIR"
  fi
  
  # 设置密钥文件路径
  KEY_PATH="$SSH_DIR/id_rsa"
  
  # 生成 SSH 密钥对
  ssh-keygen -t rsa -b 2048 -f "$KEY_PATH" -N "" -q
  
  # 将 authorized_keys 文件拷贝到 .ssh 目录
  cp shared_nfs/k85/authorized_keys $SSH_DIR
  ```

### 1.2 主节点安装 Ansible，并配置节点ip
示例文件：./ansible_workspace/inventory.ini
```ini
[hosts]
of27 ansible_host=192.168.1.27 ansible_ssh_common_args='-o StrictHostKeyChecking=no'
of25 ansible_host=192.168.1.25 ansible_ssh_common_args='-o StrictHostKeyChecking=no'
of26 ansible_host=192.168.1.26 ansible_ssh_common_args='-o StrictHostKeyChecking=no'
of28 ansible_host=192.168.1.28 ansible_ssh_common_args='-o StrictHostKeyChecking=no'
```

### 1.3 共享目录中拷贝镜像、数据集、models脚本
主要为设置docker内环境的脚本 和 启动docker内训练的脚本
设置docker内环境脚本(./tools/prepare_docker.sh)如下：
```Bash
#!/bin/bash
# 将tools视为共享目录
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install --upgrade pip
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu121


cd /workspace 
cp tools/models.tar.gz ./
tar -xvf models.tar.gz
pip install -r models/dev-requirements.txt
pip install -r models/Vision/classification/image/resnet50/requirements.txt

# 将需要使用到的脚本拷到对应文件夹下
cp tools/args_train_ddp_graph_resnet50.sh models/Vision/classification/image/resnet50/examples/
cp tools/train.sh models/Vision/classification/image/resnet50/
cp tools/profile.sh models/Vision/classification/image/resnet50/
```
启动dokcer内训练的脚本(./tools/train.sh)如下：
```Bash
# 根据使用的节点数，来判断本机是否开始训练
NUM_NODES=${1:-1}

if [ "$NODE_RANK" -lt "$NUM_NODES" ]; then
  bash examples/args_train_ddp_graph.sh "$NUM_NODES" 8 "$NODE_RANK" 192.168.1.27 /data/dataset/ImageNet/ofrecord 192 50 true python3 graph gpu 100 false '' 1
else
  echo do nothing
fi
```
其中[args_train_ddp_graph.sh](https://github.com/Oneflow-Inc/OneAutoTest/blob/main/ResNet50/args_train_ddp_graph.sh)参考自OneAutoTest仓库
### 1.4 使用ansible 在所有节点执行 docker load, docker tag命令
根据上文中inventory.ini文件依次在节点上创建docker，并将NODE_RANK写入docker的环境变量内，脚本(./ansible_workspace/set_docker.sh)内容如下：
```Bash
set -ex
if [ $# -ne 1 ]; then
  echo "Usage: $0 filename"
  exit 1
fi
host_file="$1"
num_hosts=$(wc -l < "$host_file")
docker_name="cd_test"

mapfile -t lines < "$host_file"

for (( i=1; i<${#lines[@]}; i++ )); do
  line="${lines[$i]}"
  host_name=$(echo "$line" | awk '{print $1}')
  # 根据inventory.ini文件中节点顺序，将NODE_RANK写入docker的环境变量中
  ansible $host_name -i $host_file -m shell -a "docker run -itd -e NODE_RANK=$((i-1)) -v /data/dataset/ImageNet:/data/dataset/ImageNet -v /data/home/chende/tools:/workspace/tools --network host --gpus all --shm-size=16g --ulimit memlock=-1 --ulimit core=0 --ulimit stack=67108864 --privileged --ipc host --cap-add=IPC_LOCK --name $docker_name nvcr.io/nvidia/pytorch:24.03-py3 bash"
done
# 在docker内运行环境设置的脚本
ansible hosts -i "$host_file" -m shell -a "docker exec $docker_name bash -c 'bash /workspace/tools/prepare_docker.sh'"
```
使用方式：
```Bash
bash set_docker.sh inventory.ini
```

## 第二步：进行测试

### 2.1 自动测试与日志搜集

编写一个测试命令脚本文件（./ansible_workspace/train.sh）
```Bash
#!/bin/bash
set -ex
if [ $# -ne 1 ]; then
  echo "Usage: $0 num_nodes"
  exit 1
fi
NUM_NODES="$1"
docker_name="cd_test_new"
ansible hosts -i inventory.ini -m shell -a "docker exec $docker_name bash -c 'cd /workspace/models/Vision/classification/image/resnet50 && bash train.sh $NUM_NODES'"
```

- 需要一个参数: 节点数，
- 运行该命令能够自动启动相应数量的节点运行。
- 运行结束后收集日志到主节点。
- 保存日志的目录可以以：`prefix_节点数_日期时间_surfix` 命名，前缀和后缀可以自定义

### 2.2 自动日志解析

可以使用2.1节提供的命令运行多次，比如：

```bash
train.sh 1
train.sh 2
train.sh 4
train.sh 8
train.sh 16
```

完成后应该保存了多个日志目录，需要编写一个日志处理脚本，从这些日志目录中提取性能数据并制成 markdown 格式的表格

注：不需要完整训练，训练稳定后获取到数据就可以了。

### 2.3 自动 nsys 性能测试

需要编写一个能够运行 nsys 的性能测试脚本文件（./ansible_workspace/profile.sh），和2.1的脚本类似，只是启动时需要调用nsys，我们需要搜集这些信息分析，然后进行优化。这个脚本文件。
```Bash
#!/bin/bash
set -ex
if [ $# -ne 1 ]; then
  echo "Usage: $0 num_nodes"
  exit 1
fi
NUM_NODES="$1"
docker_name="cd_test_new"
ansible hosts -i inventory.ini -m shell -a "docker exec $docker_name bash -c 'cd /workspace/models/Vision/classification/image/resnet50 && bash profile.sh $NUM_NODES'"
```
```Bash
# 根据使用的节点数，来判断是否在本地开始profile
NUM_NODES=${1:-1}

if [ "$NODE_RANK" -lt "$NUM_NODES" ]; then
  # 在启动训练时添加nsys启动路径，即可进行profile
  bash examples/args_train_ddp_graph.sh "$NUM_NODES" 8 "$NODE_RANK" 192.168.1.27 /data/dataset/ImageNet/ofrecord 192 50 true python3 graph gpu 100 false '/usr/local/cuda/bin/nsys' 1
else
  echo do nothing
fi
```
[args_train_ddp_graph.sh](https://github.com/Oneflow-Inc/OneAutoTest/blob/main/ResNet50/args_train_ddp_graph.sh)中包含使用nsys启动的选项
- 需要一个参数: 节点数，
- 运行该命令能够自动启动相应数量的节点运行。
- 运行结束后收集日志和nsys相关文件到主节点。
- 保存日志的目录可以以：`prefix_节点数_日期时间_surfix` 命名，前缀和后缀可以自定义