import os
import sys
import json

import util
import config as configs

def set_python_path():
    absolute_path = os.path.join(os.getcwd(), __file__)
    gpt2_path = os.path.dirname(os.path.dirname(absolute_path))
    sys.path.insert(0, gpt2_path)
set_python_path()

from third_party.load_dataset import load_dataset, Sampler
from third_party import encoder

from model import GPT2
import oneflow.typing as tp
import oneflow as flow

args = configs.get_args()

batch_size = args.batch_size_per_device * args.gpu_num_per_node * args.num_nodes

@flow.global_function("train", util.GetFunctionConfig(args))
def GPT2_Job(X: tp.Numpy.Placeholder((batch_size, args.seq_len), dtype=flow.int64)):
    bsz, seq_len = X.shape
    gpt2_model = GPT2(args)
    results = gpt2_model.forward(X)
    logits = results['logits']

    labels = flow.slice(X, begin=[None, 1], size=[None, seq_len-1])
    logits = flow.slice(logits, begin=[None, 0, None], size=[None, seq_len-1, None])

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=X[:, 1:], logits=logits[:, :-1])
    loss = flow.math.reduce_mean(loss, axis=-1)
    opt = util.CreateOptimizer(args)
    opt.minimize(loss)
    return {'loss': loss}


def main():
    flow.config.enable_debug_mode(True)
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)

    util.InitNodes(args)

    snapshot = util.Snapshot(args.model_save_dir, args.model_load_dir)

    print('Loading dataset...')
    enc = encoder.get_encoder(args)
    chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
    data_sampler = Sampler(chunks, seed=1)
    print('dataset has', data_sampler.total_size, 'tokens')

    metric = util.Metric(desc='train', print_steps=args.loss_print_every_n_iter,
                         batch_size=batch_size, keys=['loss'])
    print('Training...')
    try:
        for iter in range(args.iter_num):
            b = data_sampler.sample_batch(batch_size, args.seq_len)
            GPT2_Job(b).async_get(metric.metric_cb(iter))
    except KeyboardInterrupt:
        #snapshot.save("last_snapshot")
        print('interrupted')

if __name__ == '__main__':
    main()
