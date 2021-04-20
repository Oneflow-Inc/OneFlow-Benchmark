import oneflow as flow
import oneflow_api
from oneflow.python.framework.function_util import global_function_or_identity

import numpy as np
import cv2
import time
import argparse
from PIL import Image

import torch

import models.pytorch_resnet50 as pytorch_resnet50
from models.resnet50 import resnet50
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels

def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--model_path", type=str, default="./resnet50-19c8e357.pth", help="model path"
    )
    parser.add_argument(
        "--image_path", type=str, default="./data/fish.jpg", help="input image path"
    )
    return parser.parse_args()

def load_image(image_path='data/fish.jpg'):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    im = Image.open(image_path)
    im = im.resize((224, 224))
    im = im.convert('RGB')  # 有的图像是单通道的，不加转换会报错
    im = np.array(im).astype('float32')
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')

def rmse(l, r):
    return np.sqrt(np.mean(np.square(l - r)))

def main(args):
    flow.env.init()
    flow.enable_eager_execution()

    
    start_t = time.time()
    res50_module = resnet50()
    dic = res50_module.state_dict()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))


    start_t = time.time()
    torch_params = torch.load(args.model_path)
    torch_keys = torch_params.keys()

    for k in dic.keys():
        if k in torch_keys:
            dic[k] = torch_params[k].detach().numpy()

    res50_module.load_state_dict(dic)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    learning_rate = 0.01
    mom = 0
    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)

    # # set for eval mode
    # res50_module.eval()

    start_t = time.time()
    image_nd = load_image(args.image_path)
    image = flow.Tensor(image_nd, placement=flow.placement("gpu", ["0:0"], None), is_consistent=True, requires_grad=True)

    label_nd = np.array([1], dtype=np.int32) 
    label = flow.Tensor(label_nd, dtype=flow.int32, requires_grad=False)
    corss_entropy = flow.nn.CrossEntropyLossV2()
    
    bp_iters = 10
    for i in range(bp_iters):
        logits, pool, layer4, avg_pool = res50_module(image)
        loss = corss_entropy(logits, label)
        # grad = flow.Tensor(1)
        # grad = flow.Tensor(1, 2048, 1, 1)
        # grad = flow.Tensor(1, 1000)
        # flow.nn.init.ones_(grad)
        # grad.determine()
        @global_function_or_identity()
        def job():
            # logits.backward(grad)
            loss.backward()
        job()

        # for p in res50_module.parameters():
            # p[:] = p - learning_rate * p.grad
            # p[:] = p.grad

        of_sgd.step()
        of_sgd.zero_grad()

    of_loss = loss.numpy()
    of_in_grad = image.grad.numpy()
    predictions = logits.softmax()
    of_predictions = predictions.numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    clsidx = np.argmax(of_predictions)
    print("of predict prob: %f, class name: %s" % (np.max(of_predictions), clsidx_2_labels[clsidx]))
    logits_of = logits.numpy()


    #####################################################################################################
    # pytorch resnet50
    torch_res50_module = pytorch_resnet50.resnet50()
    start_t = time.time()
    torch_res50_module.load_state_dict(torch_params)
    end_t = time.time()
    print('torch load params time : {}'.format(end_t - start_t))

    # set for eval mode
    # torch_res50_module.eval()
    torch_res50_module.to('cuda')

    torch_sgd = torch.optim.SGD(torch_res50_module.parameters(), lr=learning_rate, momentum=mom)

    start_t = time.time()
    image = torch.tensor(image_nd)
    image = image.to('cuda')
    image.requires_grad = True
    corss_entropy = torch.nn.CrossEntropyLoss()
    corss_entropy.to('cuda')
    label = torch.tensor(label_nd, dtype=torch.long, requires_grad=False).to('cuda')

    for i in range(bp_iters):
        torch_sgd.zero_grad()
        logits, pool, layer4, avg_pool = torch_res50_module(image)
        loss = corss_entropy(logits, label)
        loss.backward()
        # logits.backward(torch.ones_like(logits))
        
        # for p in torch_res50_module.parameters():
            # p.data.add_(p.grad.data, alpha=-learning_rate)
            # p.data = p.grad.data

        torch_sgd.step()
        torch_sgd.zero_grad()
        
    torch_loss = loss.cpu().detach().numpy()
    torch_in_grad = image.grad.cpu().detach().numpy()
    predictions = logits.softmax(-1)
    torch_predictions = predictions.cpu().detach().numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    clsidx = np.argmax(torch_predictions)
    print("torch predict prob: %f, class name: %s" % (np.max(torch_predictions), clsidx_2_labels[clsidx]))


    # check of and torch param and grad error
    print("logit rmse error: ", rmse(logits_of, logits.cpu().detach().numpy()))
    print("prediction rmse error: ", rmse(of_predictions, torch_predictions))
    print("loss rmse error: ", rmse(of_loss, torch_loss), of_loss, torch_loss)
    print("input grad rmse error: ", rmse(torch_in_grad, of_in_grad))

    of_grads = {}
    of_params = {}
    for k, v in res50_module.named_parameters():
        of_grads[k] = v.grad.numpy() if v.grad is not None else np.random.rand(*v.numpy().shape)
        of_params[k] = v.numpy()

    for k, v in torch_res50_module.named_parameters():
        torch_grad = v.grad.cpu().detach().numpy() if v.grad is not None else np.random.rand(*v.shape)
        torch_param = v.cpu().detach().numpy()

        if k == "fc.weight":
            print("of grad:", of_grads[k].flatten()[:20])
            print("torch grad", torch_grad.flatten()[:20])

            print("of param:", of_params[k].flatten()[:20])
            print("torch param", torch_param.flatten()[:20])


        if np.allclose(of_grads[k], torch_grad, atol=1e-6) == False:
            print("of and torch grad not match, key: %s, rmse_error: %f" % (k, rmse(of_grads[k], torch_grad)))
        if np.allclose(of_params[k], torch_param, atol=1e-6) == False:
            print("of and torch param not match, key: %s, rmse_error: %f" % (k, rmse(of_params[k], torch_param)))


if __name__ == "__main__":
    args = _parse_args()
    main(args)








