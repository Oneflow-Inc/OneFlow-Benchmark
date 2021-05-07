oneflow_gpt_src_dir=${ONEFLOW_GPT_SRC_DIR:-"$PWD/oneflow_gpt"}
cache_dir=$HOME/var-cache
wheelhouse_dir=${ONEFLOW_WHEEL_PATH}
extra_mount_path=${ONEFLOW_GPT_EXTRA_MOUNT_PATH}
image=${ONEFLOW_IMAGE:-"oneflow-manylinux2014-cuda10.2:0.1"}
bash_script=${ONEFLOW_GTP_PRETRAIN_SCRIPT}
proxy=
# proxy=http://10.10.0.2:7890

docker_args=""

if [ ! -z "${proxy}" ]; then
    docker_args+=" -e http_proxy=${proxy} -e https_proxy=${proxy} -e HTTP_PROXY=${proxy} -e HTTPS_PROXY=${proxy}"
fi

if [ ! -z "${wheelhouse_dir}" ]; then
    docker_args+=" -e ONEFLOW_WHEEL_PATH=${wheelhouse_dir}"
fi

if [ ! -z "${extra_mount_path}" ]; then
    docker_args+=" -v ${extra_mount_path}:${extra_mount_path}"
fi

docker_cmd="export PATH=/opt/python/cp36-cp36m/bin:$PATH && bash ${bash_script}"

docker run -it --rm --privileged --network host --shm-size=8g \
    ${docker_args} \
    -v ${oneflow_gpt_src_dir}:${oneflow_gpt_src_dir} \
    -v cache_dir=$HOME/var-cache
    -v $PWD:$PWD \
    -w $PWD \
    --name oneflow_gpt
    ${image} bash -c "${docker_cmd}"
