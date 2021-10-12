import os
import sys
from shutil import copy,copytree
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var


def npy_compare(lhs_path, rhs_path):
    lhs = np.load(lhs_path)
    rhs = np.load(rhs_path)
    return np.allclose(lhs, rhs)


def walk_compare_npy(lhs, rhs):
    assert os.path.isdir(lhs)
    assert os.path.isdir(rhs)

    same = 0
    diff = 0
    ignore = 0
    for root, dirs, files in os.walk(lhs):
        for name in filter(lambda f: f.endswith(".npy"), files):
            lhs_path = os.path.join(root, name)
            rhs_path = os.path.join(rhs, os.path.relpath(lhs_path, lhs))
            if os.path.exists(rhs_path) and os.path.isfile(rhs_path):
                if not npy_compare(lhs_path, rhs_path):
                    print("{} False".format(lhs_path))
                    diff += 1
                else:
                    same += 1
            else:
                print("{} ignore".format(lhs_path))
                ignore += 1
    print("same:", same)
    print("diff:", diff)
    print("ignore:", ignore)


def get_varible_name(var_org):
    # for item in sys._getframe().f_locals.items():
    #     print(item[0],item[1])
    # for item in sys._getframe(1).f_locals.items():
    #     print(item[0],item[1])
    for item in sys._getframe(2).f_locals.items():
        if var_org is item[1]:
            return item[0]


def dump_to_npy(tensor, root="./output", sub="", name=""):
    if sub != "":
        root = os.path.join(root, str(sub))
    if not os.path.isdir(root):
        os.makedirs(root)

    var_org_name = get_varible_name(tensor) if name == "" else name
    path = os.path.join(root, f"{var_org_name}.npy")
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.numpy()
    np.save(path, tensor)


def save_param_npy(module, root="./output"):
    for name, param in module.named_parameters():
        # if name.endswith('bias'):
        dump_to_npy(param.numpy(), root=root, sub=0, name=name)


def param_hist(param, name, root="output"):
    print(name, param.shape)
    # print(param.flatten())

    # the histogram of the data
    n, bins, patches = plt.hist(param.flatten(), density=False, facecolor="g")

    # plt.xlabel('Smarts')
    # plt.ylabel('value')
    plt.title(f"Histogram of {name}")
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.savefig(os.path.join(root, f"{name}.png"))
    plt.close()


def save_param_hist_pngs(module, root="output"):
    for name, param in module.named_parameters():
        # if name.endswith('bias'):
        param_hist(param.numpy(), name, root=root)


def merge_param_from_old_version(src, dst):
    param_list = [
        ["deep_embedding.weight", "deep_embedding"],
        ["wide_embedding.weight", "wide_embedding"],
        ["linear_layers.fc0.features.0.bias", "fc0-bias"],
        ["linear_layers.fc0.features.0.weight", "fc0-weight"],
        ["linear_layers.fc1.features.0.bias", "fc1-bias"],
        ["linear_layers.fc1.features.0.weight", "fc1-weight"],
        ["linear_layers.fc2.features.0.bias", "fc2-bias"],
        ["linear_layers.fc2.features.0.weight", "fc2-weight"],
        ["linear_layers.fc3.features.0.bias", "fc3-bias"],
        ["linear_layers.fc3.features.0.weight", "fc3-weight"],
        ["linear_layers.fc4.features.0.bias", "fc4-bias"],
        ["linear_layers.fc4.features.0.weight", "fc4-weight"],
        ["linear_layers.fc5.features.0.bias", "fc5-bias"],
        ["linear_layers.fc5.features.0.weight", "fc5-weight"],
        ["linear_layers.fc6.features.0.bias", "fc6-bias"],
        ["linear_layers.fc6.features.0.weight", "fc6-weight"],
        ["deep_scores.weight", "deep_scores-weight"],
        ["deep_scores.bias", "deep_scores-bias"],
    ]
    for new_name, old_name in param_list:
        src_file = os.path.join(src, old_name)
        dst_file = os.path.join(dst, new_name)
        copytree(src_file, dst_file)
        print(src_file, dst_file)
    
    for new_name, old_name in param_list:
        src_file = os.path.join('/home/shiyunxiao/checkpoints/initial_checkpoint', new_name+'/meta')
        dst_file = os.path.join('/home/shiyunxiao/checkpoint_new', new_name+'/meta')
        copy(src_file, dst_file)
        print(src_file, dst_file)

if __name__ == "__main__":
    # walk_compare_npy("output/old_0", "output/0")
    merge_param_from_old_version('/home/shiyunxiao/checkpoint_old','/home/shiyunxiao/checkpoint_new')