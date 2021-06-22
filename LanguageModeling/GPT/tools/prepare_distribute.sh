#!/bin/bash

inventory=$(dirname $0)/ansible_inventory
hosts=
wksp=
oneflow_gpt_src_dir=
wheel=
pip_install=on

function help() {
    echo "Usage: prepare_distribute.sh [ -h | --help ]
                                       [ -i | --inventory inventory_file ]
                                       [ -n | --hosts hosts]
                                       [ -s | --src oneflow_gpt_src_dir ]
                                       [ -w | --wheel wheel_file ]
                                       [ --no-install ] workspace_dir"
    exit 2
}

function parse_args() {
    args=$(getopt -o hi:n:s:w: -a -l help,intentory:,hosts:,src:,wheel:,no-install -- "$@")
    if [[ $? -ne 0 ]]; then
        help
    fi

    echo "parsed args is ${args}"
    eval set -- "${args}"

    while :
    do
        case "$1" in
            -h|--help)
                echo "help"
                ;;
            -i|--intentory)
                inventory="$2"
                shift
                ;;
            -n|--hosts)
                hosts="$2"
                shift
                ;;
            -s|--src)
                oneflow_gpt_src_dir="$2"
                shift
                ;;
            -w|--wheel)
                wheel="$2"
                shift
                ;;
            --no-install)
                pip_install=
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unexpected option: $1"
                help
                ;;
        esac
        shift
    done

    echo "remaining args are: $@"
    echo "remaining args number are: $#"
    if [[ $# -ne 0 ]]; then
        wksp=$1
    else
        wksp=$PWD
    fi
}

parse_args $@

if [[ -z "${hosts}" ]]; then
    echo "hosts is unset"
    exit 1
fi

ansible ${hosts} -i ${inventory} -m file -a "path=${wksp} state=directory"

if [[ ! -z "${wheel}" ]]; then
    wheel=$(realpath "${wheel}")
    wheel_dir=$(realpath $(dirname "${wheel}"))
    ansible ${hosts} -i ${inventory} -m file -a "path=${wheel_dir} state=directory"
    ansible ${hosts} -i ${inventory} -m copy -a "src=${wheel} dest=${wheel}"
    if [[ ! -z "${pip_install}" ]]; then
        ansible ${hosts} -i ${inventory} -m shell -a "python3 -m pip install ${wheel} --user"
    fi
fi

if [[ ! -z "${oneflow_gpt_src_dir}" ]]; then
    ansible ${hosts} -i ${inventory} -m copy -a "src=${oneflow_gpt_src_dir} dest=${wksp}/oneflow_gpt"
    if [[ ! -z "${pip_install}" ]]; then
        ansible ${hosts} -i ${inventory} -m shell -a "python3 -m pip install -e ${wksp}/oneflow_gpt --user"
    fi
fi
