import argparse
import oneflow as flow
import datetime
import os
import glob
from sklearn.metrics import roc_auc_score
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, required=True)
parser.add_argument('--train_data_part_num', type=int, required=True)
parser.add_argument('--train_part_name_suffix_length', type=int, default=-1)
parser.add_argument('--train_data_num', type=int, default=36674623)
parser.add_argument('--eval_data_dir', type=str, required=True)
parser.add_argument('--eval_data_part_num', type=int, required=True)
parser.add_argument('--eval_part_name_suffix_length', type=int, default=-1)
parser.add_argument('--eval_data_num', type=int, default=4583478)
parser.add_argument('--test_data_dir', type=str, required=True)
parser.add_argument('--test_data_part_num', type=int, required=True)
parser.add_argument('--test_part_name_suffix_length', type=int, default=-1)
parser.add_argument('--test_data_num', type=int, default=4582516)
parser.add_argument('--batch_size', type=int, default=16384)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--wide_vocab_size', type=int, default=3200000)
parser.add_argument('--deep_vocab_size', type=int, default=3200000)
parser.add_argument('--deep_embedding_vec_size', type=int, default=16)
parser.add_argument('--deep_dropout_rate', type=float, default=0.5)
parser.add_argument('--num_dense_fields', type=int, default=13)
parser.add_argument('--num_wide_sparse_fields', type=int, default=2)
parser.add_argument('--num_deep_sparse_fields', type=int, default=26)
parser.add_argument('--epoch_num', type=int, default=4)
parser.add_argument('--loss_print_every_n_iter', type=int, default=100)
parser.add_argument('--gpu_num', type=int, default=8)
parser.add_argument('--hidden_units_num', type=int, default=7)
parser.add_argument('--hidden_size', type=int, default=1024)

FLAGS = parser.parse_args()

#DEEP_HIDDEN_UNITS = [1024, 1024]#, 1024, 1024, 1024, 1024, 1024]
DEEP_HIDDEN_UNITS = [FLAGS.hidden_size for i in range(FLAGS.hidden_units_num)]
print(DEEP_HIDDEN_UNITS)
train_epoch_size = FLAGS.train_data_num // FLAGS.batch_size + 1
eval_epoch_size = FLAGS.eval_data_num // FLAGS.batch_size + 1
test_epoch_size = FLAGS.test_data_num // FLAGS.batch_size + 1


def _data_loader_ofrecord(data_dir, data_part_num, batch_size, part_name_suffix_length=-1,
                          shuffle=True):
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


def _model(dense_fields, wide_sparse_fields, deep_sparse_fields):
    wide_sparse_fields = flow.parallel_cast(wide_sparse_fields, distribute=flow.distribute.broadcast())
    wide_embedding_table = flow.get_variable(
        name='wide_embedding',
        shape=(FLAGS.wide_vocab_size, 1),
        initializer=flow.random_uniform_initializer(minval=-0.05, maxval=0.05),
        distribute=flow.distribute.split(0),
    )
    wide_embedding = flow.gather(params=wide_embedding_table, indices=wide_sparse_fields)
    wide_embedding = flow.reshape(wide_embedding, shape=(-1, wide_embedding.shape[-1] * wide_embedding.shape[-2]))
    wide_scores = flow.math.reduce_sum(wide_embedding, axis=[1], keepdims=True)
    wide_scores = flow.parallel_cast(wide_scores, distribute=flow.distribute.split(0),
                                     gradient_distribute=flow.distribute.broadcast())

    deep_sparse_fields = flow.parallel_cast(deep_sparse_fields, distribute=flow.distribute.broadcast())
    deep_embedding_table = flow.get_variable(
        name='deep_embedding',
        shape=(FLAGS.deep_vocab_size, FLAGS.deep_embedding_vec_size),
        initializer=flow.random_uniform_initializer(minval=-0.05, maxval=0.05),
        distribute=flow.distribute.split(1),
    )
    deep_embedding = flow.gather(params=deep_embedding_table, indices=deep_sparse_fields)
    deep_embedding = flow.parallel_cast(deep_embedding, distribute=flow.distribute.split(0),
                                        gradient_distribute=flow.distribute.split(2))
    deep_embedding = flow.reshape(deep_embedding, shape=(-1, deep_embedding.shape[-1] * deep_embedding.shape[-2]))
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


def _get_train_conf():
    train_conf = flow.FunctionConfig()
    train_conf.default_data_type(flow.float)
    train_conf.train.primary_lr(FLAGS.learning_rate)
    train_conf.train.model_update_conf({
        'lazy_adam_conf': {
        }
    })
    train_conf.default_distribute_strategy(flow.distribute.consistent_strategy())
    train_conf.indexed_slices_optimizer_conf(dict(include_op_names=dict(op_name=['wide_embedding', 'deep_embedding'])))
    return train_conf

def _get_eval_conf():
    eval_conf = flow.FunctionConfig()
    eval_conf.default_data_type(flow.float)
    eval_conf.default_distribute_strategy(flow.distribute.consistent_strategy())
    return eval_conf


global_loss = 0.0
def _create_train_callback(epoch, step):
    def nop(loss):
        global global_loss
        global_loss += loss.mean()
        pass

    def print_loss(loss):
        global global_loss
        global_loss += loss.mean()
        print(epoch, step+1, 'time', datetime.datetime.now(), 'loss',
              global_loss/FLAGS.loss_print_every_n_iter)
        global_loss = 0.0

    if (step + 1) % FLAGS.loss_print_every_n_iter == 0:
        return print_loss
    else:
        return nop

@flow.global_function(_get_train_conf())
def train_job():
    labels, dense_fields, wide_sparse_fields, deep_sparse_fields = \
        _data_loader_ofrecord(data_dir=FLAGS.train_data_dir,
                              data_part_num=FLAGS.train_data_part_num,
                              batch_size=FLAGS.batch_size,
                              part_name_suffix_length=FLAGS.train_part_name_suffix_length,
                              shuffle=True)
    logits = _model(dense_fields, wide_sparse_fields, deep_sparse_fields)
    loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    flow.losses.add_loss(loss)
    return loss


@flow.global_function(_get_eval_conf())
def eval_job():
    labels, dense_fields, wide_sparse_fields, deep_sparse_fields = \
        _data_loader_ofrecord(data_dir=FLAGS.eval_data_dir,
                              data_part_num=FLAGS.eval_data_part_num,
                              batch_size=FLAGS.batch_size,
                              part_name_suffix_length=FLAGS.eval_part_name_suffix_length,
                              shuffle=False)
    logits = _model(dense_fields, wide_sparse_fields, deep_sparse_fields)
    loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    predict = flow.math.sigmoid(logits)
    return loss, predict, labels

@flow.global_function(_get_eval_conf())
def test_job():
    labels, dense_fields, wide_sparse_fields, deep_sparse_fields = \
        _data_loader_ofrecord(data_dir=FLAGS.test_data_dir,
                              data_part_num=FLAGS.test_data_part_num,
                              batch_size=FLAGS.batch_size,
                              part_name_suffix_length=FLAGS.test_part_name_suffix_length,
                              shuffle=False)
    logits = _model(dense_fields, wide_sparse_fields, deep_sparse_fields)
    loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    predict = flow.math.sigmoid(logits)
    return loss, predict, labels

def main():
    flow.config.gpu_device_num(FLAGS.gpu_num)
    #flow.config.enable_numa_aware_cuda_malloc_host(True)
    #flow.config.collective_boxing.enable_fusion(False)
    check_point = flow.train.CheckPoint()
    check_point.init()
    global global_loss
    for epoch in range(FLAGS.epoch_num):
        global_loss = 0.0
        for i in range(train_epoch_size):
            train_job().async_get(_create_train_callback(epoch, i))

        labels = np.array([[0]])
        preds = np.array([[0]])
        eval_loss = 0.0
        for i in range(eval_epoch_size):
            loss, pred, ref = eval_job().get()
            label_ = ref.ndarray().astype(np.float32)
            labels = np.concatenate((labels, label_), axis=0)
            preds = np.concatenate((preds, pred.ndarray()), axis=0)
            eval_loss += loss.mean()
        auc = roc_auc_score(labels[1:], preds[1:])
        print(epoch, "eval_loss", eval_loss/eval_epoch_size, "eval_auc", auc)

    labels = np.array([[0]])
    preds = np.array([[0]])
    eval_loss = 0.0
    for i in range(test_epoch_size):
        loss, pred, ref = test_job().get()
        label_ = ref.ndarray().astype(np.float32)
        labels = np.concatenate((labels, label_), axis=0)
        preds = np.concatenate((preds, pred.ndarray()), axis=0)
        eval_loss += loss.mean()
    auc = roc_auc_score(labels[1:], preds[1:])
    print("test_loss", eval_loss/test_epoch_size, "eval_auc", auc)


if __name__ == '__main__':
    main()
