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


def get_args(print_args=True):
    def str_list(x):
        return x.split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_format", type=str, default="ofrecord", help="ofrecord or onerec"
    )
    parser.add_argument(
        "--use_single_dataloader_thread",
        action="store_true",
        help="use single dataloader threads per node or not.",
    )
    parser.add_argument("--model_load_dir", type=str, default="")
    parser.add_argument("--model_save_dir", type=str, default="")
    parser.add_argument(
        "--save_initial_model",
        action="store_true",
        help="save initial model parameters or not.",
    )
    parser.add_argument("--num_dataloader_thread_per_gpu", type=int, default=2)
    parser.add_argument(
        "--data_dir", type=str, default="/dataset/wdl_ofrecord/ofrecord"
    )
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--eval_batchs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--wide_vocab_size", type=int, default=1603616)
    parser.add_argument("--deep_vocab_size", type=int, default=1603616)
    parser.add_argument("--hf_wide_vocab_size", type=int, default=800000)
    parser.add_argument("--hf_deep_vocab_size", type=int, default=800000)
    parser.add_argument("--deep_embedding_vec_size", type=int, default=16)
    parser.add_argument("--deep_dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_dense_fields", type=int, default=13)
    parser.add_argument("--num_wide_sparse_fields", type=int, default=2)
    parser.add_argument("--num_deep_sparse_fields", type=int, default=26)
    parser.add_argument("--max_iter", type=int, default=30000)
    parser.add_argument("--gpu_num_per_node", type=int, default=8)
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="node/machine number for training"
    )
    parser.add_argument(
        "--node_ips",
        type=str_list,
        default=["192.168.1.13", "192.168.1.14"],
        help='nodes ip list for training, devided by ",", length >= num_nodes',
    )
    parser.add_argument(
        "--ctrl_port", type=int, default=50051, help="ctrl_port for multinode job"
    )
    parser.add_argument("--hidden_units_num", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument(
        "--ddp", action="store_true", help="Run model in distributed data parallel mode"
    )
    parser.add_argument(
        "--execution_mode", type=str, default="eager", help="graph or eager"
    )
    parser.add_argument(
        "--test_name", type=str, default="noname_test"
    )
    

    FLAGS = parser.parse_args()
    if print_args:
        _print_args(FLAGS)
    return FLAGS


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


if __name__ == "__main__":
    get_args()
