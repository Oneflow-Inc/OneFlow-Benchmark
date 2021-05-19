import numpy as np
import pickle as pkl
import os


def compare(x, y, x_name=None, y_name=None):
    if x_name is not None and y_name is not None:
        print("****** compare {} and {} ******".format(x_name, y_name))
    else:
        print("****** compare ******")

    x[np.where(np.abs(x) <= 1e-6)] = 1e-6

    print("max abs diff: {}".format(np.max(np.abs(x - y))))
    print("mean abs diff: {}".format(np.mean(x - y)))
    print("max relative diff: {}".format(np.max(np.abs((x - y) / x))))
    print("mean relative diff: {}".format(np.mean(np.abs((x - y) / x))))
    print("\n\n")


# Forward
mx_conv0_out = np.load("/home/scxfjiang/Desktop/mx_blobs/conv0_out.npy")
of_conv0_out = np.load("/home/scxfjiang/Desktop/of_blobs/conv0_out.npy")
compare(mx_conv0_out, of_conv0_out, "mx_conv0_out", "of_conv0_out")
mx_fc_out = np.load("/home/scxfjiang/Desktop/mx_blobs/fc1_out.npy")
of_fc_out = np.load("/home/scxfjiang/Desktop/of_blobs/fc1001.npy")
compare(mx_fc_out, of_fc_out, "mx_fc_out", "of_fc_out")


# Backward
def np_load_conv_weight_grad(stage=1, unit=1, conv=1):
    conv_mx2of = {3: '2c', 2: '2b', 1: '2a', '1sc': '1'}
    mx_name = "stage{}_unit{}_conv{}_weight_grad.npy".format(stage, unit, conv)
    of_name = "res{}_{}_branch{}-weight_diff.npy".format(stage+ 1, unit - 1, conv_mx2of[conv])
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
count = 0
for stage in range(4):
    for unit in range(units[stage]):
        for conv in [1, 2, 3, '1sc']:
            if unit > 0 and conv == '1sc':
                continue
            else:
                mx, of, mx_name, of_name = np_load_conv_weight_grad(stage + 1, unit + 1, conv)
                compare(mx, of, mx_name, of_name)
                mx, of, mx_name, of_name = np_load_bn_grad(stage + 1, unit + 1, conv, 'gamma')
                compare(mx, of, mx_name, of_name)
                mx, of, mx_name, of_name = np_load_bn_grad(stage + 1, unit + 1, conv, 'beta')
                compare(mx, of, mx_name, of_name)
                count = count + 1

# MxNet Optimizer
mx_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/conv0_weight_grad.npy")
mx_fc_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/fc1_weight_grad.npy")

initialized_arg_params = pkl.load(open("/home/scxfjiang/Desktop/mxnet_params/initialized/arg_params", "rb"))
initialized_aux_params = pkl.load(open("/home/scxfjiang/Desktop/mxnet_params/initialized/aux_params", "rb"))
updated_arg_params = pkl.load(open("/home/scxfjiang/Desktop/mx_blobs/updated_arg_params", "rb"))
updated_aux_params = pkl.load(open("/home/scxfjiang/Desktop/mx_blobs/updated_aux_params", "rb"))

mx_grad = mx_fc_weight_grad / 32
init_weight = initialized_arg_params["fc1_weight"].asnumpy()
updated_weight = updated_arg_params["fc1_weight"].asnumpy()
mx_model_diff = (init_weight - updated_weight)
compare(mx_grad, mx_model_diff, "mx_grad", "mx model diff")

# OneFlow Optimizer
of_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/conv1-weight_diff.npy")
of_fc_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/fc_weight_diff.npy")

of_grad = of_fc_weight_grad / 32
init_weight = initialized_arg_params["fc1_weight"].asnumpy()
of_model_save_path = "/home/scxfjiang/repos/OneFlow-Benchmark/Classification/cnns/output/snapshots/"
with open(os.path.join(of_model_save_path, os.listdir(of_model_save_path)[0],
                       "snapshot_updated_params/Resnet-fc1001-weight/out"), "rb") as f:
    updated_weight = np.frombuffer(f.read(1000 * 2048 * 4), dtype=np.float32).reshape((1000, 2048))
of_model_diff = init_weight - updated_weight
compare(of_grad, of_model_diff, "of_grad", "of model diff")
