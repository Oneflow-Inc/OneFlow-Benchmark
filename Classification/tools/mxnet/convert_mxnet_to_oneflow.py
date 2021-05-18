import os
import numpy as np
import argparse
import pickle as pkl

import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb
import oneflow as flow
import google.protobuf.text_format as text_format

parser = argparse.ArgumentParser()

parser.add_argument("--mxnet_model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="MxNet model dir.")
parser.add_argument("--model_name_map_path",
                    default=None,
                    type=str,
                    required=True,
                    help="model_name_map.txt.")
parser.add_argument("--dump_dir",
                    default="./snapshot",
                    type=str,
                    help="dump root dir.")
parser.add_argument("--save_momentum",
                    default=False,
                    type=bool,
                    help="if extract momentum")
parser.add_argument("--name2momentum_path",
                    default=None,
                    type=str,
                    help="model_name2momentum_buffer.pkl")

args = parser.parse_args()


class ModelConverter:
    def __init__(self, mxnet_model_dir, model_name_map_path,
                 dump_dir, save_momentum=False, name2momentum_path=None):
        assert os.path.exists(model_name_map_path), "Can't find model_name_map.txt."

        model_name_map = {}
        with open(model_name_map_path) as f:
            for line in f.readlines():
                word_list = line.rstrip().replace("\t", " ").split()
                save_path = os.path.join(dump_dir, word_list[2])
                # name -> (need_trans, save_path)
                model_name_map[word_list[0]] = (True if word_list[1] == "True" else False, save_path)

        self.model_name_map = model_name_map
        self.arg_params = pkl.load(open(os.path.join(mxnet_model_dir, "arg_params"), "rb"))
        self.aux_params = pkl.load(open(os.path.join(mxnet_model_dir, "aux_params"), "rb"))
        self.model_dict = {}
        self.model_dict.update(self.arg_params)
        self.model_dict.update(self.aux_params)
        # if save_momentum:
        #     assert os.path.exists(name2momentum_path), "Can't find model_name2momentum_buffer.pkl."
        #     self.save_momentum = save_momentum
        #     name2momentum = np.load(name2momentum_path, allow_pickle=True)
        # self.name2momentum = name2momentum

    def save_model(self, model, save_path, need_trans=False):
        model = model.astype(np.float32)
        if need_trans:
            np.transpose(model)
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # write out
        f = open(save_path, 'wb')
        f.write(model.tobytes())
        f.close()
        np.save(save_path, model)

        # write meta
        meta_info = variable_meta_info_pb.VariableMetaInfo()
        meta_info.shape.dim[:] = model.shape
        meta_info.data_type = flow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(flow.float32)
        f = open(os.path.join(dir_name, "meta"), "wb")
        f.write(text_format.MessageToBytes(meta_info))
        f.close()

    def convert_model(self):
        for name, tup in self.model_name_map.items():
            if name in self.model_dict:
                model = self.model_dict[name].asnumpy()
                print("Converting {:s}, shape = {:s}".format(name, str(model.shape)))
                self.save_model(model, tup[1], tup[0])
            else:
                assert False, "No such model in PyTorch: {:s}.\n".format(name)

        print("{:d} / {:d} models are converted to OneFlow models!".format(len(self.model_dict), len(self.model_name_map)))

    def convert_momentum(self):
        pass
        # def gen_save_path(of_model_save_path):
        #     word_list = of_model_save_path.split('/')
        #     base = '/'.join(word_list[:-2]) + '/model_update-'
        #     return base + word_list[-2] + "-" + word_list[-1] + "/momentum"
        #
        # count = 0
        # for name, momentum in self.name2momentum.items():
        #     save_path = gen_save_path(self.model_name_map[name][1])
        #     print("Converting momentum to {:s}, shape = {:s}".format(save_path, str(momentum.shape)))
        #     self.save_model(momentum, save_path)
        #     count = count + 1
        #
        # print("{:d} / {:d} momentum are converted to OneFlow models!".format(count, len(self.name2momentum)))


if __name__ == "__main__":
    converter = ModelConverter(args.mxnet_model_dir,
                               args.model_name_map_path,
                               args.dump_dir,
                               args.save_momentum,
                               args.name2momentum_path)
    converter.convert_model()
    # if converter.save_momentum:
    #     converter.convert_momentum()
