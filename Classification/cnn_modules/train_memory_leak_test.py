import oneflow as flow
from oneflow.python.framework.function_util import global_function_or_identity
import numpy as np
from models.resnet50 import resnet50


flow.env.init()
flow.enable_eager_execution()

res50_module = resnet50()
# # set for eval mode
# res50_module.eval()

batch = 8
image_nd = np.ones((batch, 3, 224, 224), dtype=np.float32)
label_nd = np.array([e for e in range(batch)], dtype=np.int32)

bp_iters = 100000000

for i in range(bp_iters):
    print(i)

    # NOTE(Liang Depeng): cpu memory leak
    image = flow.Tensor(image_nd)
    n = image.numpy()[0]
    ################################
    
    # NOTE(Liang Depeng): gpu memory leak
    # label = flow.Tensor(label_nd, dtype=flow.int32, requires_grad=False)
    # logits = res50_module(image)
    
    # # uncomment following codes will solve the gpu memory leak
    # # grad = flow.Tensor(batch, 1000)
    # # grad.determine()
    # # @global_function_or_identity()
    # # def job():
    # #     logits.backward(grad)
    # # job()
    #####################









