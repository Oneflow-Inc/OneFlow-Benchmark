import numpy as np

mx_conv0_out = np.load("/home/scxfjiang/Desktop/mx_blobs/conv0_out.npy")
of_conv0_out = np.load("/home/scxfjiang/Desktop/of_blobs/conv0_out.npy")
mx_fc_out = np.load("/home/scxfjiang/Desktop/mx_blobs/fc1_out.npy")
of_fc_out = np.load("/home/scxfjiang/Desktop/of_blobs/fc1001.npy")
of_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/conv1-weight_diff.npy")
mx_conv0_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/conv0_weight_grad.npy")
of_fc_weight_grad = np.load("/home/scxfjiang/Desktop/of_blobs/grad/fc_weight_diff.npy")
mx_fc_weight_grad = np.load("/home/scxfjiang/Desktop/mx_blobs/grad/fc1_weight_grad.npy")

print(np.allclose(mx_conv0_out, of_conv0_out, rtol=1e-5, atol=1e-2))
print(np.allclose(mx_fc_out, of_fc_out, rtol=1e-5, atol=5e-2))
print(np.allclose(mx_conv0_weight_grad, of_conv0_weight_grad, rtol=8e-3, atol=2e-1))
print(np.allclose(mx_fc_weight_grad, of_fc_weight_grad, rtol=1e-5, atol=5e-2))

