#!/bin/bash
set -ex

oneflow_srd_dir=$HOME/repos/oneflow
cache_dir=$HOME/var-cache
proxy=
python_scripts_dir=$PWD/manylinux2014-build-cache/10.2/build-oneflow/python_scripts
# python_scripts_dir=$PWD/manylinux2014-build-cache/cpu/build-oneflow/python_scripts
# wheelhouse_dir=$PWD/wheel_house/oneflow_cpu-0.3b5-cp36-cp36m-manylinux2014_x86_64.whl

docker_extra_args="--shm-size 8g --cap-add SYS_PTRACE --security-opt seccomp=unconfined"
# docker_extra_args="--shm-size 8g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --gpus all"

docker_env_args=""

if [ ! -z "${proxy}" ]; then
    docker_env_args+=" -e http_proxy=${proxy} -e https_proxy=${proxy} -e HTTP_PROXY=${proxy} -e HTTPS_PROXY=${proxy}"
fi

if [ ! -z "${python_scripts_dir}" ]; then
    docker_env_args+=" -e PYTHONPATH=${python_scripts_dir}"
fi

if [ ! -z "${wheelhouse_dir}" ]; then
    docker_env_args+=" -e ONEFLOW_WHEEL_PATH=${wheelhouse_dir}"
fi

image="oneflow-manylinux2014-cuda10.2:0.1"

docker_cmd="export PATH=/opt/python/cp36-cp36m/bin:$PATH && bash"
# docker_cmd="export PATH=/opt/python/cp36-cp36m/bin:$PATH && bash ci/test/try_install.sh && bash ci/test/1node_custom_op_test.sh"
# docker_cmd="export PATH=/opt/python/cp36-cp36m/bin:$PATH && bash ci/test/1node_custom_op_test.sh"

docker run -it --rm ${docker_extra_args} \
    -v /dataset:/dataset \
    -v $HOME:$HOME \
    -v ${cache_dir}:/var/cache \
    -v ${oneflow_srd_dir}:${oneflow_srd_dir} \
    -v $PWD:$PWD \
    -w $PWD \
    ${docker_env_args} \
    --name zwx_dev \
    ${image} bash -c "${docker_cmd}"
