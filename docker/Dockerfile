ARG CUDA=10

FROM nvidia/cuda:${CUDA}.0-cudnn7-devel-centos7
RUN yum update -y && yum install -y epel-release

ARG USE_PYTHON_3_OR_2=3

RUN if [ "${USE_PYTHON_3_OR_2}" -eq 2 ] ; then yum update -y \
    && yum install -y python-devel.x86_64 \
    && curl https://bootstrap.pypa.io/get-pip.py --output ./get-pip.py \
    && python ./get-pip.py \
    && rm get-pip.py \
    && pip install numpy==1.12.0 protobuf ; fi

RUN if [ "${USE_PYTHON_3_OR_2}" -eq 3 ] ; then yum update -y \
    && yum install -y rh-python36 python36-devel.x86_64 \
    && yum install openblas-devel -y \
    && python3 -m ensurepip \
    && pip3 install numpy protobuf ; fi


## INSTALL WHEEL
RUN pip${USE_PYTHON_3_OR_2} install wheel
COPY docker/wheel/ /wheel
RUN pip${USE_PYTHON_3_OR_2} install /wheel/*.whl

