'''
将InputExample转换为oneflow的ofrecord feature
'''

import oneflow.core.record.record_pb2 as ofrecord
import six
import os
import numpy as np
import random
import struct
from typing import Tuple, List, Dict, Optional
import log
from data_utils.utils import InputFeatures, InputExample

logger = log.get_logger('root')


def int32_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int32_list=ofrecord.Int32List(value=value))

def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))

def float_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))

def double_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))

def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))

def generate_dataset(
        config,
        data: List[InputExample],
        preprocessor,
        labelled: bool = True,
        ofrecord_dir: str = None,
        stage: str = 'train',
        load_weight: bool = False # 是否为每个样本添加权重
):
    features = convert_examples_to_features(config, data, preprocessor, labelled=labelled)  # 将输入的样本（inputExample对象）进行转化为feature
    # print('len(features)=', len(features))
    # print('===============train features================')
    # print('len=', len(features))
    # print('feature 0:', features[0])
    # print('feature 1:', features[1])
    # print('===============================')
    feature_dicts = [] # list(dict)
    if not os.path.exists(os.path.join(ofrecord_dir, stage)):
        os.makedirs(os.path.join(ofrecord_dir, stage))
    fw = open(os.path.join(ofrecord_dir, stage, stage + '.of_record-0'), 'wb')

    if load_weight:
        '''
        sample_weight_dict = {idx: weight}
        array([{0: 0.9915}, {1: 0.98956}, {2: 0.9723}, {3: 0.98445}, {4: 0.99237},
        ......
        {94: 0.96635}, {95: 0.98768}], dtype=object)
        '''
        try:
            sample_weight_dict = np.load(os.path.join(ofrecord_dir, 'train/weight.npy'), allow_pickle=True)[()]
            weight_dict = dict()
            for i in sample_weight_dict:
                weight_dict.update(i)
        except:
            raise FileNotFoundError("Please run the preprocess.py to generate weight.npy at first")

    for f in features:
        weight = [1.]
        if load_weight:
            assert f.idx in weight_dict.keys()
            weight = [weight_dict[f.idx]]
            print('weight=', weight)
        feature_dict = {
            'input_ids': int32_feature(f.input_ids),
            'attention_masks': int32_feature(f.attention_mask),
            'token_type_ids': int32_feature(f.token_type_ids),
            'tasks': int32_feature(f.task),  # add by wjn
            'labels': int32_feature(f.label),
            'logits': int32_feature(f.logits),
            'idxs': int32_feature(f.idx),
            'weights': float_feature(weight)
        }
        ofrecord_features = ofrecord.OFRecord(feature=feature_dict) # 调用 ofrecord.OFRecord 创建 OFRecord 对象
        serilizedBytes = ofrecord_features.SerializeToString()  # 调用 OFRecord 对象的 SerializeToString 方法得到序列化结果
        length = ofrecord_features.ByteSize()
        fw.write(struct.pack("q", length))
        fw.write(serilizedBytes)
        feature_dicts.append(feature_dict)
    fw.close()
    return feature_dicts



def load_glue_data(path: str, stage: str):

    int32_feature_list = ['input_ids', 'attention_masks', 'token_type_ids', 'tasks',
                    'labels', 'logits', 'idxs']
    float_feature_list = []

    feature_data = dict()

    with open(os.path.join(path, stage, stage + ".of_record-0"), "rb") as f:
        while True:
            try:
                length = struct.unpack("q", f.read(8))
            except:
                break
            serilizedBytes = f.read(length[0])
            ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)

            for feature_name in int32_feature_list:
                data = ofrecord_features.feature[feature_name].int32_list.value
                if feature_name not in feature_data.keys():
                    feature_data[feature_name] = []
                feature_data[feature_name].append(data)

            for feature_name in float_feature_list:
                data = ofrecord_features.feature[feature_name].float_list.value
                if feature_name not in feature_data.keys():
                    feature_data[feature_name] = []
                feature_data[feature_name].append(data)

    for feature_name, data in feature_data.items():
        feature_data[feature_name] = np.array(data)

    return feature_data


# InputExample -> InputFeatures
def convert_examples_to_features(
        config,
        examples: List[InputExample],
        preprocessor,
        labelled: bool = True
) -> List[InputFeatures]:
    features = []
    # preprocessor = PREPROCESSORS[MLM_WRAPPER](config, config.task_name, config.pattern_id)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example {}".format(ex_index))
        # 获得input_feature。self.preprocessor根据当前的任务类型（比如MLM）获得相应的preprocessor
        input_features = preprocessor.get_input_features(example, labelled=labelled)
        features.append(input_features)
        """
        if ex_index < 5:
            logger.info(f'--- Example {ex_index} ---')
            logger.info(input_features.pretty_print(self.tokenizer))
        """
    return features