# 拉取或导入镜像

注: 用户需要有各台机器的docker权限

## 拉取镜像

适用于直接从 dockerhub 拉取镜像。

用法: `./pull.sh [镜像标签]`

参数说明:

- 镜像标签 (可选)  : 要拉取的Docker镜像标签，例如 alpine:latest。如果未提供，则使用playbook中的默认值。

示例:

- 默认使用:

```bash
./pull.sh
```

- 指定镜像标签:

 ```bash
./pull.sh alpine:latest
 ```

## 导入镜像

适用于本地共享目录有已经保存镜像的tar文件，使用 `docker load` 导入。

用法: `./load.sh [镜像文件路径] [镜像标签] [强制导入]`

参数说明:

- 镜像文件路径 (可选)  : 要导入的Docker镜像tar文件路径，默认为 `/share_nfs/k85/oneflow.0.9.1.dev20240203-cuda11.8.tar`
- 镜像标签 (可选)      : 导入后设置的Docker镜像标签，默认为 `oneflowinc/oneflow:0.9.1.dev20240203-cuda11.8`
- 强制导入 (可选)      : 是否强制导入镜像（true 或 false），默认为 false

示例:

- 默认使用:

```bash
./load.sh
```

- 指定镜像文件路径和标签:

```bash
./load.sh /path/to/shared/abc.tar myrepo/myimage:latest
```

- 强制导入镜像:    

```bash
./load.sh /path/to/shared/abc.tar myrepo/myimage:latest true
```




