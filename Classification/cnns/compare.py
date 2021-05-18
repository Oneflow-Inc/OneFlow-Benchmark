import numpy as np
import pickle as pkl

mx_conv0_out = np.load("/home/scxfjiang/Desktop/mx_blobs/conv0_out.npy")
of_conv0_out = np.load("/home/scxfjiang/Desktop/of_blobs/conv0_out.npy")
mx_fc_out = np.load("/home/scxfjiang/Desktop/mx_blobs/fc1_out.npy")
of_fc_out = np.load("/home/scxfjiang/Desktop/of_blobs/fc1001.npy")
of_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/conv1-weight_diff.npy")
mx_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/conv0_weight_grad.npy")
of_fc_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/fc_weight_diff.npy")
mx_fc_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/fc1_weight_grad.npy")

# Forward and Backward
print("compare conv0 out")
print(np.allclose(mx_conv0_out, of_conv0_out, rtol=1e-5, atol=1e-2))
print("compare fc out")
print(np.allclose(mx_fc_out, of_fc_out, rtol=1e-5, atol=5e-2))
print("compare fc weight grad")
print(np.allclose(mx_fc_weight_grad, of_fc_weight_grad, rtol=1e-5, atol=5e-2))
print("compare conv0 weight grad")
print(np.allclose(mx_conv0_weight_grad, of_conv0_weight_grad, rtol=8e-3, atol=2e-1))

# Optimizer
initialized_arg_params = pkl.load(open("/home/scxfjiang/Desktop/mxnet_params/initialized/arg_params", "rb"))
initialized_aux_params = pkl.load(open("/home/scxfjiang/Desktop/mxnet_params/initialized/aux_params", "rb"))
updated_arg_params = pkl.load(open("/home/scxfjiang/Desktop/mx_blobs/updated_arg_params", "rb"))
updated_aux_params = pkl.load(open("/home/scxfjiang/Desktop/mx_blobs/updated_aux_params", "rb"))

grad = mx_fc_weight_grad / 32
init_weight = initialized_arg_params["fc1_weight"].asnumpy()
updated_weight = updated_arg_params["fc1_weight"].asnumpy()
model_diff = (init_weight - updated_weight)
print(np.mean(model_diff - grad))
print(np.max(np.abs(model_diff - grad)))
print(model_diff - grad)
print(np.argwhere(np.abs(model_diff - grad) > 0.1).shape)
print(np.argwhere(np.abs(model_diff - grad) <= 0.1).shape)
