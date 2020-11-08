import os
import oneflow as flow

def CreateOptimizer(args):
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [args.learning_rate])
    if args.optimizer == 'adam':
        opt = flow.optimizer.Adam(lr_scheduler, do_bias_correction=True) 
    elif args.optimizer == 'sgd':
        opt = flow.optimizer.SGD(lr_scheduler, momentum=0.0)
    else:
        exit('Bad optimizer:', args.optimizer)
    return opt


def GetFunctionConfig(args):
    config = flow.function_config()
    #config.enable_auto_mixed_precision(args.use_fp16)
    #if args.use_xla:
    #    config.use_xla_jit(True)
    config.enable_fuse_add_to_output(True)
    config.enable_fuse_model_update_ops(True)
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
            self.save('initial_model')
            print("Init model on demand.")

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        self._check_point.save(snapshot_save_path)
