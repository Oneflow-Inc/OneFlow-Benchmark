# run_dist_training.sh 使用说明

`run_dist_training.sh` 是一个 Bash 脚本，用于运行 `ansible-playbook` 命令来启动分布式训练。此脚本支持通过参数指定 Docker 镜像和源目录。

## 用法

```bash
./run_dist_training.sh [docker_image] [src]
```

## 参数

- `docker_image` (可选): 要使用的 Docker 镜像名称。默认为 `oneflowinc/oneflow:0.9.1.dev20240203-cuda11.8`。
- `src` (可选): 要挂载到 Docker 容器的源目录。默认为 `/share_nfs/k85/models/Vision/classification/image/resnet50`。

## 示例

1. 使用默认值运行：

```bash
./run_dist_training.sh
```

2. 指定 Docker 镜像运行：

```bash
./run_dist_training.sh "my_custom_image:latest"
```

3. 指定 Docker 镜像和源目录运行：

```bash
./run_dist_training.sh "my_custom_image:latest" "/my/custom/src"
```

## 注意

如果不提供参数，脚本将使用默认的 Docker 镜像和源目录。

