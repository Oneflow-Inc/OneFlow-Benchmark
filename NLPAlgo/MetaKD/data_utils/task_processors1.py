# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 运行命令
# 本文件目标：生成训练集各个domain class对应的prototype embedding，并计算每个样本的prototypical score
# python3 preprocess.py --task_name g1 --data_dir data/k-shot-cross/g1/16-42 --num_epochs 4 --seed 42 --seq_length=128 --train_example_num 96  --eval_example_num 96 --vocab_file uncased_L-12_H-768_A-12/vocab.txt --resave_ofrecord

"""
This file contains the logic for loading data for all tasks.
"""

import csv
import sys
import uuid
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
from tqdm import tqdm
import struct
import six
import oneflow as flow
import oneflow.core.record.record_pb2 as ofrecord
import numpy as np
import log
# from pet import task_helpers
from data_utils.utils import domain_to_id, label_map
# from transformers import DataProcessor as TransDataProcessor


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



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, domain=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain = domain


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, idxs, input_ids, input_mask, segment_ids, weights, example, task_name='senti'):
        self.idxs = idxs
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.domain = domain_to_id[task_name][example.domain]
        self.label = label_map[task_name][example.label]
        self.weights = weights
        self.example = example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_examples(self, data_path, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_path, 'train.tsv')), "train", domain)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, domain=None):
        """Creates examples for the training and dev sets."""
        examples = []
        cnt = 0
        domain_list = domain.split(",") if domain else None
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if domain and line[3] not in domain_list:
                continue
            if cnt == 0:
                print(line[0], line[1], line[2], line[3], line[8], line[9], line[-1])
                cnt += 1
            guid = line[2]
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, domain=line[3]))
        return examples


class SentiProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_examples(self, data_path, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_path, 'train.tsv')), "train", domain)

    def get_labels(self):
        """See base class."""
        return ["positive", "negative"]

    def _create_examples(self, lines, set_type, genre=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 3:
                import pdb
                pdb.set_trace()
            review, domain, sentiment = line
            if genre and genre != "mix" and domain != genre:
                continue
            guid = uuid.uuid4()
            text_a = review
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=sentiment, domain=domain))
        return examples



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(args, examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 1:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: {}".format(example.label))

        features.append(
            InputFeatures(idxs=ex_index,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          weights=1.0,
                          example=example,
                          task_name=args.task_name)
        )
    return features



def generate_dataset(
        config,
        data: List[InputExample],
        tokenizer,
        labelled: bool = True,
        ofrecord_dir: str = None,
        stage: str = 'train',
        load_weight: bool = False # 是否为每个样本添加权重
):
    features = convert_examples_to_features(config, data, config.max_seq_length, tokenizer)  # 将输入的样本（inputExample对象）进行转化为feature
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
            assert f.idxs in weight_dict.keys()
            weight = [weight_dict[f.idxs]]
            # print('weight=', weight)
        feature_dict = {
            'input_ids': int32_feature(f.input_ids),
            'input_mask': int32_feature(f.input_mask),
            'segment_ids': int32_feature(f.segment_ids),
            'domain': int32_feature(f.domain),  # add by wjn
            'label': int32_feature(f.label),
            'idxs': int32_feature(f.idxs),
            'weights': float_feature(weight)
        }
        # feature_dict = {
        #     'input_ids': int32_feature(f.input_ids),
        #     'input_mask': int32_feature(f.input_mask),
        #     'segment_ids': int32_feature(f.segment_ids),
        # }
        ofrecord_features = ofrecord.OFRecord(feature=feature_dict) # 调用 ofrecord.OFRecord 创建 OFRecord 对象
        serilizedBytes = ofrecord_features.SerializeToString()  # 调用 OFRecord 对象的 SerializeToString 方法得到序列化结果
        length = ofrecord_features.ByteSize()
        fw.write(struct.pack("q", length))
        fw.write(serilizedBytes)
        feature_dicts.append(feature_dict)
    fw.close()
    return feature_dicts



processors = {
    "mnli": MnliProcessor,
    "senti": SentiProcessor
}