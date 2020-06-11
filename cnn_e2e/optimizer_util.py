from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

def add_optimizer_args(parser):
    group = parser.add_argument_group('optimizer parameters',
                                      'entire group applies only to optimizer parameters')
    group.add_argument("--model_update", type=str, default="sgd", help="sgd, adam, momentum")
    group.add_argument("--learning_rate", type=float, default=0.256)
    group.add_argument("--wd", type=float, default=1.0/32768, help="weight decay")
    group.add_argument("--mom", type=float, default=0.875, help="momentum")
    group.add_argument('--lr_decay', type=str, default='cosine', help='cosine, step, polynomial, None')
    group.add_argument('--warmup_epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    return parser

def gen_model_update_conf(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    epoch_size = math.ceil(args.num_examples / train_batch_size)
    num_train_batches = epoch_size * args.num_epochs
    num_warmup_batches = epoch_size * args.warmup_epochs
    decay_batches = num_train_batches - num_warmup_batches

    model_update_conf = {}
    # basic model update
    if args.model_update == 'sgd':
        model_update_conf["naive_conf"] = {}
    elif args.model_update == 'adam':
        model_update_conf["adam_conf"] = {"beta1": 0.9}
    elif args.model_update == 'momentum':
        assert args.mom < 1.0 
        assert args.mom > 0.0
        model_update_conf["momentum_conf"] = {"beta": args.mom}
    else:
        assert False

    # learning rate warmup
    if args.warmup_epochs > 0: #linear warmup only
        model_update_conf['warmup_conf'] = {"linear_conf": {
            "warmup_batches": num_warmup_batches, 
            "start_multiplier": 0,
        }}

    # learning rate decay
    if args.lr_decay == 'cosine':
        model_update_conf['learning_rate_decay'] = {"cosine_conf": {"decay_batches": decay_batches}}
    elif args.lr_decay == 'step':
        boundaries = [x * epoch_size for x in [30, 60, 80]] 
        scales = [1, 0.1, 0.01, 0.001]
        model_update_conf['learning_rate_decay'] = {"piecewise_scaling_conf": {
            "boundaries": boundaries, 
            "scales":scales,
        }}
    elif args.lr_decay == 'polynomial':
        model_update_conf['learning_rate_decay'] = {"polynomial_conf": {
            "decay_batches": decay_batches, 
            "end_learning_rate": 0.00001,
        }}
    
    # weight decay
    # if args.wd > 0:
    #     assert args.wd < 1.0
    #     model_update_conf['weight_decay_conf'] = {
    #         "weight_decay_rate": args.wd, 
    #         "excludes": {"pattern": ['_bn-']}
    #     }

    import pprint
    pprint.pprint(model_update_conf)
    return model_update_conf


if __name__ == '__main__':
    import config as configs
    parser = configs.get_parser()
    args = parser.parse_args()
    configs.print_args(args)
    gen_model_update_conf(args)
