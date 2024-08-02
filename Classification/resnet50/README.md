# 使用Ansible在多节点环境分布式训练

文件目录

```
.
├── 0_dist_ssh_key                    # 分发 SSH 公钥到各个节点
│   ├── distribute_ssh_key.yml        # ansible playbook
│   ├── dist_ssh_key.sh               # 执行脚本
│   ├── inventory.ini                 # 仅用于分发公钥的主机清单文件，需要根据实际情况配置
│   ├── README.md                     # 说明文件
│   └── vars.yml                      # 初始未加密的用户密码文件，需经过配置并加密后使用
├── 1_get_docker_image                # 各个节点获取 docker 镜像
│   ├── load_and_tag_docker_image.yml # 导入镜像 ansible playbook
│   ├── load.sh                       # 导入镜像执行脚本
│   ├── pull_docker_image.yml         # 拉取镜像 ansible playbook
│   ├── pull.sh                       # 拉取镜像执行脚本
│   └── README.md                     # 说明文件
├── 2_distributed_training            # 分布式训练
│   ├── dist_training.yml             # 用于分布式训练的 ansible playbook 
│   └── README.md                     # 说明文件
│   └── run_dist_training.sh          # 分布式训练执行脚本
├── inventory.ini                     # 主机清单文件，需要根据实际情况配置
└── README.md                         # 说明文件
```




