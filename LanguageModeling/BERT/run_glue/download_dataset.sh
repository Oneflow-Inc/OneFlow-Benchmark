#!/usr/bin/env bash

echo ========download the dataset with ofrecord format=======

wget -P ./ https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/glue_data_bin.zip
unzip -d ./ ./glue_data_bin.zip