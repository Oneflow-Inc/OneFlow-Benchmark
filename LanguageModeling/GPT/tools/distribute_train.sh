ansible 192.168.1.15 -i ~/wksp/ansible_playground/dev_nodes.yaml -m file -a "path=$PWD state=directory"

ansible 192.168.1.15 -i ~/wksp/ansible_playground/dev_nodes.yaml -m file -a "path=$PWD/wheelhouse state=directory"

ansible 192.168.1.15 -i ~/wksp/ansible_playground/dev_nodes.yaml -m copy -a "src=$PWD/wheelhouse/oneflow-0.3.5+cu102.git.ad01890ff-cp36-cp36m-manylinux2014_x86_64.whl dest=$PWD/wheelhouse/oneflow-0.3.5+cu102.git.ad01890ff-cp36-cp36m-manylinux2014_x86_64.whl"

ansible localhost:192.168.1.15 -i ~/wksp/ansible_playground/dev_nodes.yaml -m copy -a "src=/home/zhangwenxiao/repos/OneFlow-Benchmark-GPT-Private/LanguageModeling/GPT/ dest=$PWD/oneflow_gpt"
