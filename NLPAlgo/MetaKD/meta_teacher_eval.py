# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
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


# 本文件用于生成训练集的prototype embedding，并计算每个样本的prototypical score
# 执行命令：python3 preprocess.py --task_name g1 --data_dir data/k-shot-cross/g1/16-42 --num_epochs 4 --seed 42 --seq_length=128 --train_example_num 96  --eval_example_num 96 --vocab_file uncased_L-12_H-768_A-12/vocab.txt --resave_ofrecord


import sys

sys.path.append("../..")
import os
import numpy as np
import math
from tqdm import tqdm
import oneflow as flow
from classifier import MetaTeacherBERT
import tokenization
from util import Snapshot, InitNodes, Metric, CreateOptimizer, GetFunctionConfig
# from data_utils.data_process import Preprocessor
from data_utils.task_processors1 import processors, convert_examples_to_features, generate_dataset
from data_utils.utils import domain_list, domain_to_id, label_map

# from data_utils.data_process import domain_list, class_list, task_to_id
import scipy.spatial.distance as distance
import config as configs

parser = configs.get_parser()
parser.add_argument("--task_name", type=str, default='senti', required=True, help="The name of the task to train.")
# parser.add_argument("--domain", type=str, default='all', required=True, help="The domain of given model.")
# parser.add_argument("--use_domain_loss", action='store_true', help="Whether to use domain loss")
# parser.add_argument('--data_portion', type=float, default=1.0, help='How many data selected.')
# parser.add_argument("--domain_loss_weight", default=0.2, type=float, help="The loss weight of domain.")
# parser.add_argument("--use_sample_weights",default=False, type=bool, help="The loss weight of domain.")
# parser.add_argument("--input", default="./inputs.tsv", type=str, required=True, help="bert embedding path")
# parser.add_argument("--output", default="./output.tsv", type=str, required=True, help="bert embedding path")
parser.add_argument("--vocab_file", type=str, default=None)
parser.add_argument("--train_data_prefix", type=str, default='train.of_record-')
parser.add_argument("--eval_data_prefix", type=str, default='eval.of_record-')
parser.add_argument("--dev_data_prefix", type=str, default='dev.of_record-')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--label_list", type=list, default=None)
parser.add_argument("--max_seq_length", default=128, type=int, required=False,
                    help="The max sequence length of input sentences.")
parser.add_argument("--batch_size_per_device", default=128, type=int, required=False, help="training batch size")
parser.add_argument("--train_data_part_num", type=int, default=1, help="data part number in dataset")
parser.add_argument("--eval_data_part_num", type=int, default=1, help="data part number in dataset")
parser.add_argument("--eval_batch_size_per_device", default=1, type=int, required=False,
                    help="evalutation batch size")
parser.add_argument("--train_example_num", default=6480, type=int, required=False,
                    help="The number of training data that need to calculate weighted values")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument("--resave_ofrecord", action='store_true', help="Whether to resave the data to ofrecord")
args = parser.parse_args()

args.num_nodes = 1  # 默认只有一个设备
args.gpu_num_per_node = 1
batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device
# epoch_size = math.ceil(args.train_example_num / batch_size)
num_eval_steps = math.ceil(args.train_example_num / eval_batch_size)


def BertDecoder(
        data_dir, batch_size, data_part_num, seq_length, part_name_prefix, shuffle=True
):
    with flow.scope.placement("cpu", "0:0"):
        # 使用ofrecord读取数据
        ofrecord = flow.data.ofrecord_reader(data_dir,
                                             batch_size=1,
                                             data_part_num=data_part_num,
                                             part_name_prefix=part_name_prefix,
                                             random_shuffle=shuffle,
                                             shuffle_after_epoch=shuffle)
        blob_confs = {}

        def _blob_conf(name, shape, dtype=flow.int32):
            # 获得标签
            blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

        _blob_conf("input_ids", [seq_length])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        _blob_conf("domain", [1])
        _blob_conf("label", [1])
        _blob_conf("idxs", [1])
        _blob_conf("weights", [1], dtype=flow.float32)
        return blob_confs


# 跑一个batch
def BuildBert(
        batch_size,
        data_part_num,
        data_dir,
        part_name_prefix,
        shuffle=True
):
    hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4
    num_domains = len(domain_list[args.task_name])
    # 获得一批数据
    decoders = BertDecoder(
        data_dir, batch_size, data_part_num, args.seq_length, part_name_prefix, shuffle=shuffle
    )
    # is_real_example = decoders['is_real_example']
    # 使用带有分类器的BERT进行微调，并获得loss和logit
    _, logit_blob, all_attention_scores_blob, pooled_output = MetaTeacherBERT(
        decoders['input_ids'],
        decoders['input_mask'],
        decoders['segment_ids'],
        decoders['label'],
        decoders['domain'],
        args.vocab_size,
        input_weight=decoders['weights'],
        num_domains=num_domains,
        seq_length=args.seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=0.02,
        get_output=False,  # 只获得隐向量
        get_att_reps=True, # 获得包括attention、hidden layers以及soft-label等信息
    )
    return  logit_blob, all_attention_scores_blob, pooled_output, decoders['idxs']


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalValJob():
    logit_blob, all_attention_scores_blob, pooled_output, idxs = BuildBert(
        batch_size,
        args.train_data_part_num,
        os.path.join(args.data_dir, 'ofrecord', 'train'),
        args.train_data_prefix,
        shuffle=False
    )
    return logit_blob, all_attention_scores_blob, pooled_output, idxs


def get_hidden_embedding(eval_job_func, num_steps, desc='train'):
    sample_to_blob = dict()
    for index in tqdm(range(num_steps)):
        logit_blob, all_attention_scores_blob, pooled_output, idxs = eval_job_func().get()
        current_size = logit_blob.shape[0]
        logit_blob = list(logit_blob.numpy().tolist())
        pooled_output = list(pooled_output.numpy().tolist())
        idxs = list(idxs.numpy().tolist())
        # print('idxs=', len(idxs))
        # print('logit_blob=', len(logit_blob))
        # print('all_attention_scores_blob=', len(all_attention_scores_blob))
        # print('pooled_output=', len(pooled_output))
        for ei, idx in enumerate(idxs):
            sample_to_blob[idx[0]] = [logit_blob[ei], all_attention_scores_blob[ei], pooled_output[ei]]

    return sample_to_blob


# 计算 prototypical score
def compute_weight(domain, label, current_embedding, centroid_embeddings):
    key_name = domain + "\t" + str(label)
    current_centroid = centroid_embeddings[key_name]
    other_centroids = list()
    for current_key in centroid_embeddings.keys():
        items = current_key.split("\t")
        current_domain = items[0]
        current_label = items[1]
        if not (current_domain == domain) and (current_label == label):
            other_centroids.append(centroid_embeddings[current_key])
    other_centroids = np.array(other_centroids)
    other_centroid_mean = np.mean(other_centroids, axis=0)
    first_cos_sim = 1 - distance.cosine(current_embedding, current_centroid)
    second_cos_sim = 1 - distance.cosine(current_embedding, other_centroid_mean)
    return (first_cos_sim + second_cos_sim) / 2


def main():
    processor = processors[args.task_name]()
    args.label_list = processor.get_labels()

    # 获得BERT分词工具
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file)

    # 将原始数据集生成ofrecord数据集
    ofrecord_dir = os.path.join(args.data_dir, 'ofrecord')

    if not os.path.exists(ofrecord_dir) or args.resave_ofrecord:
        # 获得训练集、验证集和测试集的example格式数据 List[InputExample]
        train_examples = processor.get_examples(args.data_dir)
        print('===============================')
        print('len=', len(train_examples))
        print('===============================')
        # 为input examples生成input feature，并生成ofrecord格式的数据
        train_feature_dict = generate_dataset(args, train_examples, tokenizer, ofrecord_dir=ofrecord_dir,
                                              stage='train')

    # 加载预训练模型参数
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)
    InitNodes(args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    # 在训练集上使用训练好的meta-teacher再次进行一次推理，得到每个样本对应的hidden embedding、soft label以及attention score等
    sample_to_blob = get_hidden_embedding(
        eval_job_func=BertGlueEvalValJob,
        num_steps=num_eval_steps,
        desc='eval')

    np.save(os.path.join(ofrecord_dir, 'train/sample_to_blob.npy'), sample_to_blob, allow_pickle=True)


if __name__ == "__main__":
    main()
