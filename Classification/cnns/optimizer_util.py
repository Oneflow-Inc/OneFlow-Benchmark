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
import oneflow as flow
import math
import pprint

def add_optimizer_args(parser):
    group = parser.add_argument_group('optimizer parameters',
                                      'entire group applies only to optimizer parameters') 
    group.add_argument("--optimizer", type=str, default="sgd", help="sgd, adam, rmsprop")
    group.add_argument("--learning_rate", type=float, default=0.256)
    group.add_argument("--wd", type=float, default=1.0/32768, help="weight decay")
    group.add_argument("--momentum", type=float, default=0.875, help="momentum")
    group.add_argument('--lr_decay', type=str, default='cosine', help='cosine, step, polynomial, exponential, None')
    group.add_argument('--lr_decay_rate', type=float, default='0.94', help='exponential learning decay rate')
    group.add_argument('--lr_decay_epochs', type=int, default=2, help='exponential learning rate decay every n epochs')
    group.add_argument('--warmup_epochs', type=int, default=5,
                       help='the epochs to warmp-up lr to scaled large-batch value')
    group.add_argument('--decay_rate', type=float, default='0.9', help='decay rate of RMSProp')
    group.add_argument('--epsilon', type=float, default='1', help='epsilon')
    group.add_argument('--gradient_clipping', type=float, default=0.0, help='gradient clipping')
    return parser

def set_up_optimizer(loss, args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    batches_per_epoch = math.ceil(args.num_examples / train_batch_size)
    warmup_batches = batches_per_epoch * args.warmup_epochs
    num_train_batches = batches_per_epoch * args.num_epochs
    decay_batches = num_train_batches - warmup_batches
    exponential_decay_batches = batches_per_epoch * args.lr_decay_epochs

    # set up warmup strategy
    warmup = flow.optimizer.warmup.linear(warmup_batches, 0) if warmup_batches > 0 else None
   
    # set up grad_clipping
    grad_clipping = flow.optimizer.grad_clipping.by_global_norm(args.gradient_clipping) if  args.gradient_clipping > 0.0  else None

   # set up learning rate scheduler
    if args.lr_decay == 'cosine':
        # CosineScheduler
        lr_scheduler = flow.optimizer.CosineScheduler(
            base_lr=args.learning_rate, 
            steps = decay_batches, 
            warmup=warmup
        )
    elif args.lr_decay == 'step':
        # PiecewiseScalingScheduler
        lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(
            base_lr=args.learning_rate, 
            boundaries=[30, 60, 80], 
            scale=[0.1, 0.01, 0.001], 
            warmup=warmup
        )
    elif args.lr_decay == 'polynomial':
        # PolynomialSchduler
        lr_scheduler = flow.optimizer.PolynomialSchduler(
            base_lr=args.learning_rate,
            steps=decay_batches, 
            end_learning_rate=0.00001,
            power=1.0,
            cycle=False,
            warmup=warmup        
        )
    elif args.lr_decay == 'exponential':
        # ExponentialScheduler
        lr_scheduler = flow.optimizer.ExponentialScheduler(
            base_lr=args.learning_rate,
            steps=exponential_decay_batches, 
            decay_rate=args.lr_decay_rate,
            staircase=False,
            warmup=warmup
        )
    else:
        lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(
            base_lr=args.learning_rate, 
            boundaries=[args.num_epochs], 
            scale=[1.0], 
            warmup=warmup
        )

    
    # set up optimizer
    loss_scale_policy = None
    if args.use_fp16:
        loss_scale_policy = flow.optimizer.loss_scale.dynamic_loss_scale(increment_period=2000);
    if args.optimizer=='sgd':
        print("Optimizer:  SGD")
        flow.optimizer.SGD(lr_scheduler,
            momentum=args.momentum if args.momentum>0 else None,
            grad_clipping = grad_clipping,
            loss_scale_policy=loss_scale_policy
        ).minimize(loss)
    elif args.optimizer=='adam':
        if args.wd > 0 and args.wd < 1.0 :
            print("Optimizer:  AdamW")
            flow.optimizer.AdamW(
                lr_scheduler = lr_scheduler,
                weight_decay = args.wd,
                weight_decay_excludes='_bn-',
                grad_clipping = grad_clipping,
                epsilon=args.epsilon,
                loss_scale_policy=loss_scale_policy
            ).minimize(loss)
        else:
            print("Optimizer:  Adam")
            flow.optimizer.Adam(lr_scheduler=lr_scheduler,
                grad_clipping=grad_clipping,
                epsilon=args.epsilon,
                loss_scale_policy=loss_scale_policy
            ).minimize(loss)
    elif args.optimizer=='rmsprop':
        print("Optimizer:  RMSProp")
        flow.optimizer.RMSProp(lr_scheduler=lr_scheduler,
            decay_rate=args.decay_rate,
            epsilon=args.epsilon,
            loss_scale_policy=loss_scale_policy
        ).minimize(loss)


if __name__ == '__main__':
    import config as configs
    parser = configs.get_parser()
    args = parser.parse_args()
    configs.print_args(args) 

    