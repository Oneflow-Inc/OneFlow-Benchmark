# 使用Ansible在多节点环境分布式训练

## 目录结构

```
.
├── 0_dist_ssh_key                    # 分发 SSH 公钥到各个节点
│   ├── distribute_ssh_key.yml        # Ansible playbook
│   ├── dist_ssh_key.sh               # 执行脚本
│   ├── inventory.ini                 # 仅用于分发公钥的主机清单文件，需要根据实际情况配置
│   ├── README.md                     # 说明文件
│   └── vars.yml                      # 初始未加密的用户密码文件，需经过配置并加密后使用
├── 1_get_docker_image                # 各个节点获取 Docker 镜像
│   ├── load_and_tag_docker_image.yml # 导入镜像 Ansible playbook
│   ├── load.sh                       # 导入镜像执行脚本
│   ├── pull_docker_image.yml         # 拉取镜像 Ansible playbook
│   ├── pull.sh                       # 拉取镜像执行脚本
│   └── README.md                     # 说明文件
├── 2_distributed_training            # 分布式训练
│   ├── dist_training.yml             # 用于分布式训练的 Ansible playbook
│   ├── run_dist_training.sh          # 分布式训练执行脚本
│   └── README.md                     # 说明文件
├── 3_1node_training                  # 在一个节点上训练，用于获得基准
│   ├── one_node_training.yml         # 单节点训练 Ansible playbook
│   ├── run_one_node_training.sh      # 单节点训练执行脚本
│   └── README.md                     # 说明文件
├── 4_profiling                       # 使用 nsys 采集性能相关信息
│   ├── profiling.yml                 # Profiling Ansible playbook
│   ├── run_profiling.sh              # 采集信息执行脚本
│   └── README.md                     # 说明文件
├── inventory.ini                     # 主机清单文件，需要根据实际情况配置
└── README.md                         # 说明文件
```

## 分步说明

### 0_dist_ssh_key

该目录用于分发 SSH 公钥到各个节点。

- `distribute_ssh_key.yml`: Ansible playbook，用于分发公钥。
- `dist_ssh_key.sh`: 执行分发公钥的脚本。
- `inventory.ini`: 主机清单文件，需要根据实际情况配置。
- `vars.yml`: 初始未加密的用户密码文件，需经过配置并加密后使用。

### 1_get_docker_image

该目录用于在各个节点上获取 Docker 镜像。

- `load_and_tag_docker_image.yml`: Ansible playbook，用于导入 Docker 镜像并设置标签。
- `load.sh`: 导入镜像执行脚本。
- `pull_docker_image.yml`: Ansible playbook，用于拉取 Docker 镜像。
- `pull.sh`: 拉取镜像执行脚本。

### 2_distributed_training

该目录用于执行分布式训练。

- `dist_training.yml`: 用于分布式训练的 Ansible playbook。
- `run_dist_training.sh`: 分布式训练执行脚本。

### 3_1node_training

该目录用于在一个节点上训练，以获得基准。

- `one_node_training.yml`: 单节点训练 Ansible playbook。
- `run_one_node_training.sh`: 单节点训练执行脚本。

### 4_profiling

该目录用于使用 nsys 采集性能相关信息。

- `profiling.yml`: Profiling Ansible playbook。
- `run_profiling.sh`: 采集信息执行脚本。

## 使用方法

1. **分发 SSH 公钥**:
```sh
cd 0_dist_ssh_key
./dist_ssh_key.sh
```

2. **获取 Docker 镜像**:
```sh
cd 1_get_docker_image
./pull.sh  # 或者 ./load.sh
```

3. **执行分布式训练**:
```sh
cd 2_distributed_training
./run_dist_training.sh [docker_image] [src]
```

4. **在一个节点上训练**:
```sh
cd 3_1node_training
./run_one_node_training.sh
```

5. **采集性能相关信息**:
```sh
cd 4_profiling
./run_profiling.sh
```

注意：在运行这些脚本之前，请确保已经正确配置了 `inventory.ini` 文件中的主机信息。
