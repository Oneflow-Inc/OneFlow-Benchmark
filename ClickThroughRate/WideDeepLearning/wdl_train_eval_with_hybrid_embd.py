"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import oneflow as flow
import datetime
import os
import glob
from sklearn.metrics import roc_auc_score
import numpy as np
import time

def str_list(x):
    return x.split(',')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_format', type=str, default='ofrecord', help='ofrecord or onerec')
parser.add_argument('--train_data_dir', type=str, default='')
parser.add_argument('--train_data_part_num', type=int, default=1)
parser.add_argument('--train_part_name_suffix_length', type=int, default=-1)
parser.add_argument('--eval_data_dir', type=str, default='')
parser.add_argument('--eval_data_part_num', type=int, default=1)
parser.add_argument('--eval_part_name_suffix_length', type=int, default=-1)
parser.add_argument('--eval_batchs', type=int, default=20)
parser.add_argument('--eval_interval', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=16384)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--wide_vocab_size', type=int, default=3200000)
parser.add_argument('--deep_vocab_size', type=int, default=3200000)
parser.add_argument('--hf_wide_vocab_size', type=int, default=1600000)
parser.add_argument('--hf_deep_vocab_size', type=int, default=1600000)
parser.add_argument('--deep_embedding_vec_size', type=int, default=16)
parser.add_argument('--deep_dropout_rate', type=float, default=0.5)
parser.add_argument('--num_dense_fields', type=int, default=13)
parser.add_argument('--num_wide_sparse_fields', type=int, default=2)
parser.add_argument('--num_deep_sparse_fields', type=int, default=26)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--loss_print_every_n_iter', type=int, default=100)
parser.add_argument('--gpu_num_per_node', type=int, default=8)
parser.add_argument('--num_nodes', type=int, default=1,
                    help='node/machine number for training')
parser.add_argument('--node_ips', type=str_list, default=['192.168.1.13', '192.168.1.14'],
                    help='nodes ip list for training, devided by ",", length >= num_nodes')
parser.add_argument("--ctrl_port", type=int, default=50051, help='ctrl_port for multinode job')
parser.add_argument('--hidden_units_num', type=int, default=7)
parser.add_argument('--hidden_size', type=int, default=1024)

FLAGS = parser.parse_args()

#DEEP_HIDDEN_UNITS = [1024, 1024]#, 1024, 1024, 1024, 1024, 1024]
DEEP_HIDDEN_UNITS = [FLAGS.hidden_size for i in range(FLAGS.hidden_units_num)]

def _data_loader(data_dir, data_part_num, batch_size, part_name_suffix_length=-1, shuffle=True):
    if FLAGS.dataset_format == 'ofrecord':
        return _data_loader_ofrecord(data_dir, data_part_num, batch_size, part_name_suffix_length,
                                     shuffle) 
    elif FLAGS.dataset_format == 'onerec':
        return _data_loader_onerec(data_dir, batch_size, shuffle)
    elif FLAGS.dataset_format == 'synthetic':
        return _data_loader_synthetic(batch_size)
    else:
        assert 0, "Please specify dataset_type as `ofrecord`, `onerec` or `synthetic`."
        
    

def _data_loader_ofrecord(data_dir, data_part_num, batch_size, part_name_suffix_length=-1,
                          shuffle=True):
    assert data_dir
    print('load ofrecord data form', data_dir)
    ofrecord = flow.data.ofrecord_reader(data_dir,
                                         batch_size=batch_size,
                                         data_part_num=data_part_num,
                                         part_name_suffix_length=part_name_suffix_length,
                                         random_shuffle=shuffle,
                                         shuffle_after_epoch=shuffle)
    def _blob_decoder(bn, shape, dtype=flow.int32):
        return flow.data.OFRecordRawDecoder(ofrecord, bn, shape=shape, dtype=dtype)
    labels = _blob_decoder("labels", (1,))
    dense_fields = _blob_decoder("dense_fields", (FLAGS.num_dense_fields,), flow.float)
    wide_sparse_fields = _blob_decoder("wide_sparse_fields", (FLAGS.num_wide_sparse_fields,))
    deep_sparse_fields = _blob_decoder("deep_sparse_fields", (FLAGS.num_deep_sparse_fields,))
    return flow.identity_n([labels, dense_fields, wide_sparse_fields, deep_sparse_fields])


def _data_loader_synthetic(batch_size):
    devices = ['{}:0-{}'.format(i, FLAGS.gpu_num_per_node - 1) for i in range(FLAGS.num_nodes)]
    with flow.scope.placement("cpu", devices):
        def _blob_random(shape, dtype=flow.int32, initializer=flow.zeros_initializer(flow.int32)):
            return flow.data.decode_random(shape=shape, dtype=dtype, batch_size=batch_size, 
                                           initializer=initializer)
        labels = _blob_random((1,), initializer=flow.random_uniform_initializer(dtype=flow.int32))
        dense_fields = _blob_random((FLAGS.num_dense_fields,), dtype=flow.float, 
                                    initializer=flow.random_uniform_initializer())
        wide_sparse_fields = _blob_random((FLAGS.num_wide_sparse_fields,))
        deep_sparse_fields = _blob_random((FLAGS.num_deep_sparse_fields,))
        print('use synthetic data')
    return flow.identity_n([labels, dense_fields, wide_sparse_fields, deep_sparse_fields])


def _data_loader_onerec(data_dir, batch_size, shuffle):
    assert data_dir
    print('load onerec data form', data_dir)
    files = glob.glob(os.path.join(data_dir, '*.onerec'))
    readdata = flow.data.onerec_reader(files=files, batch_size=batch_size, random_shuffle=shuffle,
                                       verify_example=False,
                                       shuffle_buffer_size=64,
                                       shuffle_after_epoch=shuffle)

    def _blob_decoder(bn, shape, dtype=flow.int32):
        return flow.data.onerec_decoder(readdata, key=bn, shape=shape, dtype=dtype)

    labels = _blob_decoder('labels', shape=(1,))
    dense_fields = _blob_decoder("dense_fields", (FLAGS.num_dense_fields,), flow.float)
    wide_sparse_fields = _blob_decoder("wide_sparse_fields", (FLAGS.num_wide_sparse_fields,))
    deep_sparse_fields = _blob_decoder("deep_sparse_fields", (FLAGS.num_deep_sparse_fields,))
    return flow.identity_n([labels, dense_fields, wide_sparse_fields, deep_sparse_fields])


def _hybrid_embedding(name, ids, embedding_size, vocab_size, hf_vocab_size):
    b, s = ids.shape
    ids = flow.flatten(ids)
    unique_ids, unique_ids_idx, _, _ = flow.experimental.unique_with_counts(ids)
    hf_vocab_size_constant = flow.constant(hf_vocab_size, dtype=flow.int32)
    hf_indices = flow.argwhere(flow.math.less(unique_ids, hf_vocab_size_constant))
    lf_indices = flow.argwhere(flow.math.greater_equal(unique_ids, hf_vocab_size_constant))
    hf_ids = flow.gather_nd(params=unique_ids, indices=hf_indices)
    lf_ids = flow.gather_nd(params=unique_ids, indices=lf_indices)
    hf_embedding_table = flow.get_variable(
            name=f'hf_{name}',
            shape=(hf_vocab_size, embedding_size),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(minval=-0.05, maxval=0.05),
            )
    hf_embedding = flow.gather(params=hf_embedding_table, indices=hf_ids)#, no_duplicates_in_indices=True)
    lf_ids = lf_ids - hf_vocab_size_constant
    with flow.scope.placement('cpu', '0:0'):
        lf_embedding_table = flow.get_variable(
                name=f'lf_{name}',
                shape=(vocab_size - hf_vocab_size, embedding_size),
                #shape=(vocab_size, embedding_size),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-0.05, maxval=0.05),
                )
        lf_embedding = flow.gather(params=lf_embedding_table, indices=lf_ids)#, no_duplicates_in_indices=True)
    unique_embedding = flow.reshape(flow.zeros_like(unique_ids, dtype=flow.float), (-1, 1)) * flow.constant(0.0, dtype=flow.float, shape=(1,embedding_size))
    # unique_embedding = flow.constant(0.0, dtype=flow.float, shape=(b*s, embedding_size))
    unique_embedding = flow.tensor_scatter_nd_update(params=unique_embedding, updates=hf_embedding, indices=hf_indices)
    unique_embedding = flow.tensor_scatter_nd_update(params=unique_embedding, updates=lf_embedding, indices=lf_indices)
    unique_embedding = flow.gather(params=unique_embedding, indices=unique_ids_idx)
    unique_embedding = flow.cast_to_static_shape(unique_embedding)
    unique_embedding = flow.reshape(unique_embedding, shape=(b, s*embedding_size))
    return unique_embedding


def _embedding(name, ids, embedding_size, vocab_size, split_axis=0):
    ids = flow.parallel_cast(ids, distribute=flow.distribute.broadcast())
    params = flow.get_variable(
        name=name,
        shape=(vocab_size, embedding_size),
        initializer=flow.random_uniform_initializer(minval=-0.05, maxval=0.05),
        distribute=flow.distribute.split(split_axis),
    )
    embedding = flow.gather(params=params, indices=ids)
    embedding = flow.reshape(embedding, shape=(-1, embedding.shape[-1] * embedding.shape[-2]))
    return embedding


def _model(dense_fields, wide_sparse_fields, deep_sparse_fields):
    # wide_embedding = _embedding('wide_embedding', wide_sparse_fields, 1, FLAGS.wide_vocab_size)
    wide_embedding = _hybrid_embedding('wide_embedding', wide_sparse_fields, 1, FLAGS.wide_vocab_size, 
                                       FLAGS.hf_wide_vocab_size)
    wide_scores = flow.math.reduce_sum(wide_embedding, axis=[1], keepdims=True)
    wide_scores = flow.parallel_cast(wide_scores, distribute=flow.distribute.split(0),
                                     gradient_distribute=flow.distribute.broadcast())

    # deep_embedding = _embedding('deep_embedding', deep_sparse_fields, FLAGS.deep_embedding_vec_size, 
    #                             FLAGS.deep_vocab_size, split_axis=1)
    deep_embedding = _hybrid_embedding('deep_embedding', deep_sparse_fields, FLAGS.deep_embedding_vec_size,
                                       FLAGS.deep_vocab_size, FLAGS.hf_deep_vocab_size)
    deep_features = flow.concat([deep_embedding, dense_fields], axis=1)
    for idx, units in enumerate(DEEP_HIDDEN_UNITS):
        deep_features = flow.layers.dense(
            deep_features,
            units=units,
            kernel_initializer=flow.glorot_uniform_initializer(),
            bias_initializer=flow.constant_initializer(0.0),
            activation=flow.math.relu,
            name='fc' + str(idx + 1)
        )
        deep_features = flow.nn.dropout(deep_features, rate=FLAGS.deep_dropout_rate)
    deep_scores = flow.layers.dense(
        deep_features,
        units=1,
        kernel_initializer=flow.glorot_uniform_initializer(),
        bias_initializer=flow.constant_initializer(0.0),
        name='fc' + str(len(DEEP_HIDDEN_UNITS) + 1)
    )

    scores = wide_scores + deep_scores
    return scores


global_loss = 0.0
def _create_train_callback(step):
    def nop(loss):
        global global_loss
        global_loss += loss.mean()
        pass

    def print_loss(loss):
        global global_loss
        global_loss += loss.mean()
        print(step+1, 'time', time.time(), 'loss',  global_loss/FLAGS.loss_print_every_n_iter)
        global_loss = 0.0

    if (step + 1) % FLAGS.loss_print_every_n_iter == 0:
        return print_loss
    else:
        return nop


def CreateOptimizer(args):
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [args.learning_rate])
    return flow.optimizer.LazyAdam(lr_scheduler)


def _get_train_conf():
    train_conf = flow.FunctionConfig()
    train_conf.default_data_type(flow.float)
    indexed_slices_ops = [
        'wide_embedding', 
        'deep_embedding', 
        'hf_wide_embedding', 
        'hf_deep_embedding',
        'lf_wide_embedding', 
        'lf_deep_embedding',
    ]
    train_conf.indexed_slices_optimizer_conf(dict(include_op_names=dict(op_name=indexed_slices_ops)))
    return train_conf


@flow.global_function('train', _get_train_conf())
def train_job():
    labels, dense_fields, wide_sparse_fields, deep_sparse_fields = \
        _data_loader(data_dir=FLAGS.train_data_dir, data_part_num=FLAGS.train_data_part_num, 
                     batch_size=FLAGS.batch_size, 
                     part_name_suffix_length=FLAGS.train_part_name_suffix_length, shuffle=True)
    logits = _model(dense_fields, wide_sparse_fields, deep_sparse_fields)
    loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    opt = CreateOptimizer(FLAGS)
    opt.minimize(loss)
    loss = flow.math.reduce_mean(loss)
    return loss


@flow.global_function()
def eval_job():
    labels, dense_fields, wide_sparse_fields, deep_sparse_fields = \
        _data_loader(data_dir=FLAGS.eval_data_dir, data_part_num=FLAGS.eval_data_part_num,
                     batch_size=FLAGS.batch_size,
                     part_name_suffix_length=FLAGS.eval_part_name_suffix_length, shuffle=False)
    logits = _model(dense_fields, wide_sparse_fields, deep_sparse_fields)
    loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    predict = flow.math.sigmoid(logits)
    return loss, predict, labels


def InitNodes(args):
    if args.num_nodes > 1:
        assert args.num_nodes <= len(args.node_ips)
        flow.env.ctrl_port(args.ctrl_port)
        nodes = []
        for ip in args.node_ips[:args.num_nodes]:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)

def print_args(args):
    print("=".ljust(66, "="))
    print("Running {}: num_gpu_per_node = {}, num_nodes = {}.".format(
        'OneFlow-WDL', args.gpu_num_per_node, args.num_nodes))
    print("=".ljust(66, "="))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("-".ljust(66, "-"))
    #print("Time stamp: {}".format(
    #    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))))


def main():
    print_args(FLAGS)
    InitNodes(FLAGS)
    flow.config.gpu_device_num(FLAGS.gpu_num_per_node)
    flow.config.enable_model_io_v2(True)
    flow.config.enable_debug_mode(True)
    flow.config.collective_boxing.nccl_enable_all_to_all(True)
    #flow.config.enable_numa_aware_cuda_malloc_host(True)
    #flow.config.collective_boxing.enable_fusion(False)
    check_point = flow.train.CheckPoint()
    check_point.init()
    for i in range(FLAGS.max_iter):
        train_job().async_get(_create_train_callback(i))
        if (i + 1 ) % FLAGS.eval_interval == 0:
            labels = np.array([[0]])
            preds = np.array([[0]])
            cur_time = time.time()
            eval_loss = 0.0
            for j in range(FLAGS.eval_batchs):
                loss, pred, ref = eval_job().get()
                label_ = ref.numpy().astype(np.float32)
                labels = np.concatenate((labels, label_), axis=0)
                preds = np.concatenate((preds, pred.numpy()), axis=0)
                eval_loss += loss.mean()
            auc = roc_auc_score(labels[1:], preds[1:])
            print(i+1, "eval_loss", eval_loss/FLAGS.eval_batchs, "eval_auc", auc)


if __name__ == '__main__':
    main()
