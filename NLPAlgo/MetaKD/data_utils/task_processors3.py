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
    def __init__(self, portion=1.0):
        self.data_portion = portion

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
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

    def get_train_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_with_weights.tsv")), "train", domain)

    def get_dev_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched", domain)

    def get_test_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
            "test_matched", domain)

    def get_aug_examples(self, data_dir, domain=None):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug", domain)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, select_domains=None):
        """Creates examples for the training and dev sets."""
        examples = []
        cnt = 0
        domain_list = select_domains.split(",") if select_domains else None
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "train":
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                domain = line[4]
                weight = line[5]
                if cnt == 0:
                    print(line)
                    cnt += 1
            else:
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
                domain = line[3]
                weight = 1.0
                if cnt == 0:
                    print(line[0], line[1], line[2], line[3], line[8], line[9], line[-1])
                    cnt += 1
            if select_domains and domain not in domain_list:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, domain=domain))

        random.shuffle(examples)
        if set_type == "train":
            return examples[:int(len(examples) * self.data_portion)]
        else:
            return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched", domain)



class SentiProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", domain)

    def get_dev_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", domain)

    def get_test_examples(self, data_dir, domain=None):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", domain)

    def get_labels(self):
        """See base class."""
        return ["positive", "negative"]

    def _create_examples(self, lines, set_type, select_domains=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        cnt = 0
        domain_list = select_domains.split(",") if select_domains else None
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # if set_type == "train":
            #     guid, text_a, _, sentiment, domain, weight = line
            #     if select_domains and domain not in domain_list:
            #         continue
            #     if cnt == 0:
            #         print(line)
            #         cnt += 1
            # else:
            text_a, domain, sentiment = line
            if select_domains and domain not in domain_list:
                continue
            guid = "%s-%s" % (set_type, line[0])
            weight = 1.0
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=sentiment, domain=domain))
        return examples


def convert_examples_to_features(args, examples, max_seq_length, tokenizer, output_mode: str='classification'):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

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

        if output_mode == "classification":
            label_id = label_map[args.task_name][example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        domain_id = domain_to_id[args.task_name][example.domain]

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))
            logger.info("domain_id: {}".format(domain_id))

        features.append(
            InputFeatures(idxs=ex_index,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          weights=1.0,
                          example=example,
                          task_name=args.task_name)
        )
        '''
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
        '''
    return features



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
            sample_to_blob = np.load(os.path.join(ofrecord_dir, 'train/sample_to_blob.npy'), allow_pickle=True)[()]
            # print('len(sample_to_blob)=', sample_to_blob)
            sample_weight_dict = np.load(os.path.join(ofrecord_dir, 'train/weight.npy'), allow_pickle=True)[()]
            # print('len(sample_weight_dict)=', sample_weight_dict)
            weight_dict = dict()
            for i in sample_weight_dict:
                weight_dict.update(i)
        except:
            raise FileNotFoundError("Please run the preprocess.py to generate weight.npy at first")

    for f in features:
        weight = [1.]
        logit_blob = [0.]
        all_attention_scores_blob = [0.]
        pooled_output = [0.]
        if load_weight:
            # 加载每个样本对应meta-teacher的各个参数
            assert f.idxs in sample_to_blob.keys()
            # [logit_blob[ei], all_attention_scores_blob[ei], pooled_output[ei]]
            logit_blob = sample_to_blob[f.idxs][0]
            all_attention_scores_blob = sample_to_blob[f.idxs][1].tolist()
            all_attention_scores_blob = np.array(all_attention_scores_blob).reshape(-1).tolist()
            pooled_output = sample_to_blob[f.idxs][2]
            pooled_output = np.array(pooled_output).reshape(-1).tolist()
            weight = [weight_dict[f.idxs]]
            # print('weight=', weight)
        # print('isinstance(all_attention_scores_blob, (list, tuple))=', isinstance(all_attention_scores_blob, (list, tuple)))
        feature_dict = {
            'input_ids': int32_feature(f.input_ids),
            'input_mask': int32_feature(f.input_mask),
            'segment_ids': int32_feature(f.segment_ids),
            'domain': int32_feature(f.domain),  # add by wjn
            'label': int32_feature(f.label),
            'idxs': int32_feature(f.idxs),
            'weights': float_feature(weight),
            'logit_blob': float_feature(logit_blob),
            'all_attention_scores_blob': float_feature(all_attention_scores_blob),
            'pooled_output': float_feature(pooled_output),
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