import os
import time
from collections import OrderedDict
import oneflow as flow

def CreateOptimizer(args):
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [args.learning_rate])
    loss_scale_policy = flow.optimizer.loss_scale.dynamic_loss_scale(increment_period=20) if args.use_fp16 else None
    #loss_scale_policy = None
    if args.optimizer == 'adam':
        opt = flow.optimizer.Adam(lr_scheduler, do_bias_correction=True, loss_scale_policy=loss_scale_policy)
    elif args.optimizer == 'sgd':
        opt = flow.optimizer.SGD(lr_scheduler, momentum=0.0)
    elif args.optimizer == 'adamw':
        opt = flow.optimizer.AdamW(lr_scheduler, do_bias_correction=True, loss_scale_policy=loss_scale_policy,
                                   beta1=args.adam_beta1, beta2=args.adam_beta2, epsilon=args.adam_eps, 
                                   weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
                                   weight_decay=args.weight_decay,
                                   grad_clipping=flow.optimizer.grad_clipping.by_global_norm(args.clip_grad))
    else:
        exit('Bad optimizer:', args.optimizer)
    return opt


def GetFunctionConfig(args):
    config = flow.function_config()
    #config.enable_auto_mixed_precision(args.use_fp16)
    #if args.use_xla:
    #    config.use_xla_jit(True)
    if args.use_fp16:
        config.enable_auto_mixed_precision(True)
    config.enable_fuse_add_to_output(True)
    config.prune_parallel_cast_ops(True)
    config.enable_fuse_model_update_ops(True)
    # turn on the flag of none-distributed-optimizer
    config.enable_non_distributed_optimizer(False)
    return config


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

class Snapshot_new(object):
    def __init__(self, model_save_dir, model_load_dir):
        self._model_save_dir = model_save_dir
        if model_load_dir:
            assert os.path.isdir(model_load_dir)
            print("Restoring model from {}.".format(model_load_dir))
            #self._check_point.load(model_load_dir)
            flow.load_variables(flow.checkpoint.get(model_load_dir))
        #else:
        #    self._check_point.init()
        #    #self.save('initial_model')
        #    print("Init model on demand.")
class Snapshot(object):
    def __init__(self, model_save_dir, model_load_dir):
        self._model_save_dir = model_save_dir
        self._check_point = flow.train.CheckPoint()
        if model_load_dir:
            assert os.path.isdir(model_load_dir)
            print("Restoring model from {}.".format(model_load_dir))
            self._check_point.load(model_load_dir)
        else:
            self._check_point.init()
            #self.save('initial_model')
            print("Init model on demand.")

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        self._check_point.save(snapshot_save_path)

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        flow.checkpoint.save(snapshot_save_path)


class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


class Metric(object):
    def __init__(self, desc='train', print_steps=-1, batch_size=256, keys=[]):
        r"""accumulate and calculate metric

        Args:
            desc: `str` general description of the metric to show
            print_steps: `Int` print metrics every nth steps
            batch_size: `Int` batch size per step
            keys: keys in callback outputs
        Returns:
        """
        self.desc = desc
        self.print_steps = print_steps
        assert batch_size > 0
        self.batch_size = batch_size

        assert isinstance(keys, (list, tuple))
        self.keys = keys
        self.metric_dict = OrderedDict()
        self.metric_dict['step'] = 0

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

    def _clear(self):
        for key in self.keys:
            self.metric_dict[key] = 0.0
        self.metric_dict['throughput'] = 0.0
        self.num_samples = 0.0

    def update_and_save(self, key, value, step, **kwargs):
        self.metric_dict[key] = value

    def metric_cb(self, step=0, **kwargs):
        def callback(outputs):
            if step == 0: self._clear()

            for key in self.keys:
                self.metric_dict[key] += outputs[key].sum()

            self.num_samples += self.batch_size

            if (step + 1) % self.print_steps == 0:
                self.metric_dict['step'] = step + 1
                for k, v in kwargs.items():
                    self.metric_dict[k] = v
                throughput = self.num_samples / self.timer.split()
                self.update_and_save('throughput', throughput, step)
                for key in self.keys:
                    value = self.metric_dict[key] / self.num_samples
                    self.update_and_save(key, value, step, **kwargs)
                print(', '.join(('{}: {}' if type(v) is int else '{}: {:.3f}').format(k, v) \
                                for k, v in self.metric_dict.items()), time.time())
                self._clear()

        return callback
