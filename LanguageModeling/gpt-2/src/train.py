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

parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

@flow.global_function("train", util.GetFunctionConfig(args))
def GPT2_Job(X: tp.Numpy.Placeholder((args.batch_size, args.seq_len), dtype=flow.int64)):
    bsz, seq_len = X.shape
    gpt2_model = GPT2(args)
    results = gpt2_model.forward(X)
    logits = results['logits']
    
    labels = flow.slice(X, begin=[None, 1], size=[None, seq_len-1])
    logits = flow.slice(logits, begin=[None, 0, None], size=[None, seq_len-1, None])

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=X[:, 1:], logits=logits[:, :-1])
    loss = flow.math.reduce_mean(loss)
    opt = util.CreateOptimizer(args)
    opt.minimize(loss)
    return {'loss': loss}


def main():
    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)

    util.InitNodes(args)

    snapshot = util.Snapshot(args.model_save_dir, args.model_load_dir)

    print('Loading dataset...')
    enc = encoder.get_encoder(args)
    chunks = load_dataset(enc, args.dataset, args.combine, encoding=args.encoding)
    data_sampler = Sampler(chunks, seed=1)
    print('dataset has', data_sampler.total_size, 'tokens')

    print('Training...')
    try:
        for iter in range(100):
            b = data_sampler.sample_batch(args.batch_size, args.seq_len)         
            output = GPT2_Job(b).get()
            print(iter, output['loss'].numpy())
    except KeyboardInterrupt:
        snapshot.save("last_snapshot")
        print('interrupted')

if __name__ == '__main__':
    main()
