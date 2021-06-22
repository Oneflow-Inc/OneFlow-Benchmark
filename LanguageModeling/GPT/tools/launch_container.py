import argparse
import os
import subprocess
import tempfile


def pwd():
    return os.getcwd()


def homepath(relative_path=None):
    if relative_path is None:
        return os.path.expanduser("~")

    return os.path.expanduser(f"~/{relative_path}")


def py_bin_path(py_ver):
    py_ver_list = py_ver.split(".")
    major, minor = py_ver_list[:2]
    ver = f"{major}{minor}"
    return f"/opt/python/cp{ver}-cp{ver}m/bin"


def launch_oneflow_gpt_container(
    cmd,
    src,
    image,
    wheel,
    extra_mount=None,
    py_ver="3.7",
    proxy=None,
    interactive=True,
    name="oneflow_gpt",
):
    bash_script = f"""set -ex
export PATH={py_bin_path(py_ver)}:$PATH
python3 -m pip install {wheel}
python3 -m pip install -e {src}
{cmd or 'bash'}
"""

    docker_args = ""

    if proxy is not None:
        docker_args += f" -e http_proxy={proxy} -e https_proxy={proxy} -e HTTP_PROXY={proxy} -e HTTPS_PROXY={proxy}"

    if extra_mount is not None:
        docker_args += f" -v {extra_mount}:{extra_mount}"

    docker_cmd = "docker run"

    if interactive:
        docker_cmd += " -it"

    docker_cmd += " --rm"
    docker_cmd += " --runtime nvidia"
    docker_cmd += " --privileged"
    docker_cmd += " --network host"
    docker_cmd += " --shm-size=8g"

    docker_cmd += docker_args
    docker_cmd += f" -v {src}:{src}"
    docker_cmd += f" -v {homepath('var-cache')}:/var/cache"
    docker_cmd += " -v /tmp:/host/tmp"
    docker_cmd += f" -v {pwd()}:{pwd()}"
    docker_cmd += f" -w {pwd()}"
    docker_cmd += f" --name {name}"
    docker_cmd += f" {image}"

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
        t_fname = f.name
        f.write(bash_script)
        f.flush()
        print("tempfile name:", t_fname)
        docker_cmd += f" bash /host{t_fname}"
        print(docker_cmd)
        subprocess.check_call(docker_cmd, shell=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default=None, help="")
    parser.add_argument("--src", type=str, default=f"{pwd()}/oneflow_gpt", help="")
    parser.add_argument(
        "--image", type=str, default="oneflow-manylinux2014-cuda11.2:0.1", help="",
    )
    parser.add_argument(
        "--wheel",
        type=str,
        default="$PWD/packages/oneflow-0.3.5+cu112.git.4a4f032-cp37-cp37m-linux_x86_64.whl",
        help="",
    )
    parser.add_argument("--extra-mount", type=str, default="/data", help="")
    parser.add_argument("--py", type=str, default="3.7", help="")
    parser.add_argument("--proxy", type=str, default=None, help="")
    parser.add_argument("--no-interactive", action="store_false", dest="interactive", help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_oneflow_gpt_container(
        args.cmd,
        args.src,
        args.image,
        args.wheel,
        args.extra_mount,
        args.py,
        args.proxy,
        args.interactive,
    )
