import numpy as np
import pickle as pkl
import os
import csv


def compare(x, y, x_name=None, y_name=None, csv_path=None, blob_name=None):
    if x_name is not None and y_name is not None:
        print("****** compare {} and {} ******".format(x_name, y_name))
    else:
        print("****** compare ******")

    x[np.where(np.abs(x) <= 1e-8)] = 1e-8

    max_abs_diff = np.max(np.abs(x - y))
    mean_abs_diff = np.mean(np.abs(x - y))
    max_relative_abs_diff = np.max(np.abs((x - y) / x))
    mean_relative_abs_diff = np.mean(np.abs((x - y) / x))

    print("max abs diff: {}".format(max_abs_diff))
    print("mean abs diff: {}".format(mean_abs_diff))
    print("max relative diff: {}".format(max_relative_abs_diff))
    print("mean relative diff: {}".format(mean_relative_abs_diff))
    print("\n\n")

    if csv_path is not None:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'blob name': blob_name, 'max abs diff': max_abs_diff, 'mean abs diff': mean_abs_diff,
                             'max relative diff': max_relative_abs_diff,
                             'mean relative diff': mean_relative_abs_diff})


# Forward
forward_csv_path = '/home/scxfjiang/Desktop/forward.csv'
if os.path.exists(forward_csv_path):
    os.system("rm {}".format(forward_csv_path))
with open(forward_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['blob name', 'max abs diff', 'mean abs diff', 'max relative diff', 'mean relative diff']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

mx_conv0_out = np.load("/home/scxfjiang/Desktop/mx_blobs/conv0_out.npy")
of_conv0_out = np.load("/home/scxfjiang/Desktop/of_blobs/conv0_out.npy")
compare(mx_conv0_out, of_conv0_out, "mx_conv0_out", "of_conv0_out", forward_csv_path, "conv0_out")
mx_fc_out = np.load("/home/scxfjiang/Desktop/mx_blobs/fc1_out.npy")
of_fc_out = np.load("/home/scxfjiang/Desktop/of_blobs/fc1001.npy")
compare(mx_fc_out, of_fc_out, "mx_fc_out", "of_fc_out", forward_csv_path, "fc_out")

# Backward
backward_csv_path = '/home/scxfjiang/Desktop/backward.csv'
if os.path.exists(backward_csv_path):
    os.system("rm {}".format(backward_csv_path))
with open(backward_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['blob name', 'max abs diff', 'mean abs diff', 'max relative diff', 'mean relative diff']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


def np_load_conv_weight_grad(stage=1, unit=1, conv=1):
    conv_mx2of = {3: '2c', 2: '2b', 1: '2a', '1sc': '1'}
    mx_name = "stage{}_unit{}_conv{}_weight_grad.npy".format(stage, unit, conv)
    of_name = "res{}_{}_branch{}-weight_diff.npy".format(stage + 1, unit - 1, conv_mx2of[conv])
    mx = np.load(os.path.join("/home/scxfjiang/Desktop/mx_blobs/grad/", mx_name)).astype(np.float32)
    of = np.load(os.path.join("/home/scxfjiang/Desktop/of_blobs/grad/", of_name)).astype(np.float32)
    return mx, of, mx_name, of_name


def np_load_bn_grad(stage=1, unit=1, bn=1, weight='gamma'):
    bn_mx2of = {3: '2c_bn_add_relu', 2: '2b_bn_relu', 1: '2a_bn_relu', '1sc': '1_bn'}
    bn_mx2mx = {3: 'bn3', 2: 'conv3', 1: 'conv2', '1sc': 'bn_sc'}
    mx_name = "stage{}_unit{}_{}_{}_grad.npy".format(stage, unit, bn_mx2mx[bn], weight)
    of_name = "res{}_{}_branch{}-{}_diff.npy".format(stage + 1, unit - 1, bn_mx2of[bn], weight)
    mx = np.load(os.path.join("/home/scxfjiang/Desktop/mx_blobs/grad/", mx_name)).astype(np.float32)
    of = np.load(os.path.join("/home/scxfjiang/Desktop/of_blobs/grad/", of_name)).astype(np.float32)
    return mx, of, mx_name, of_name


units = [3, 4, 6, 3]
for stage in range(4):
    for unit in range(units[stage]):
        for conv in [1, 2, 3, '1sc']:
            if unit > 0 and conv == '1sc':
                continue
            else:
                mx, of, mx_name, of_name = np_load_conv_weight_grad(stage + 1, unit + 1, conv)
                compare(mx, of, mx_name, of_name, backward_csv_path, of_name)
                mx, of, mx_name, of_name = np_load_bn_grad(stage + 1, unit + 1, conv, 'gamma')
                compare(mx, of, mx_name, of_name, backward_csv_path, of_name)
                mx, of, mx_name, of_name = np_load_bn_grad(stage + 1, unit + 1, conv, 'beta')
                compare(mx, of, mx_name, of_name, backward_csv_path, of_name)

mx_fc_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/fc1_weight_grad.npy")
of_fc_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/fc1001-weight_diff.npy")
compare(mx_fc_weight_grad, of_fc_weight_grad, "mx_fc_weight_grad", "of_fc_weight_grad", backward_csv_path,
        "fc_weight_grad")

# Optimizer
optimizer_csv_path = '/home/scxfjiang/Desktop/optimizer.csv'
if os.path.exists(optimizer_csv_path):
    os.system("rm {}".format(optimizer_csv_path))
with open(optimizer_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['blob name', 'max abs diff', 'mean abs diff', 'max relative diff', 'mean relative diff']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# MxNet Optimizer
mx_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/conv0_weight_grad.npy")
initialized_arg_params = pkl.load(open("/home/scxfjiang/Desktop/mxnet_params/initialized/arg_params", "rb"))
initialized_aux_params = pkl.load(open("/home/scxfjiang/Desktop/mxnet_params/initialized/aux_params", "rb"))
updated_arg_params = pkl.load(open("/home/scxfjiang/Desktop/mx_blobs/updated_arg_params", "rb"))
updated_aux_params = pkl.load(open("/home/scxfjiang/Desktop/mx_blobs/updated_aux_params", "rb"))

mx_grad = mx_fc_weight_grad / 32
init_weight = initialized_arg_params["fc1_weight"].asnumpy()
updated_weight = updated_arg_params["fc1_weight"].asnumpy()
mx_model_diff = (init_weight - updated_weight)
# compare(mx_grad, mx_model_diff, "mx_grad", "mx model diff")

# OneFlow Optimizer
of_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/conv1-weight_diff.npy")
of_grad = of_fc_weight_grad / 32
init_weight = initialized_arg_params["fc1_weight"].asnumpy()
of_model_save_path = "/home/scxfjiang/repos/OneFlow-Benchmark/Classification/cnns/output/snapshots/"
with open(os.path.join(of_model_save_path, os.listdir(of_model_save_path)[0],
                       "snapshot_updated_params/Resnet-fc1001-weight/out"), "rb") as f:
    updated_weight = np.frombuffer(f.read(1000 * 2048 * 4), dtype=np.float32).reshape((1000, 2048))
of_model_diff = init_weight - updated_weight
# compare(of_grad, of_model_diff, "of_grad", "of model diff")
compare(mx_model_diff, of_model_diff, "mx fc weight diff", "of fc weight diff", optimizer_csv_path, "fc weight diff")
print(mx_model_diff)
print(of_model_diff)

# print("*" * 100)
# model_norm = np.linalg.norm(init_weight)
# model_diff_norm = np.linalg.norm(of_grad)
# print("model_norm")
# print(model_norm)
# print("grad norm")
# print(model_diff_norm)
# lars = 1.0
# if model_norm > 0 and model_diff_norm > 0:
#     lars = 0.001 * model_norm / (0.0 + model_diff_norm)
# local_lr = 1.0 * lars
# print("local lr")
# print(local_lr)
# np_model_diff = of_grad * local_lr
