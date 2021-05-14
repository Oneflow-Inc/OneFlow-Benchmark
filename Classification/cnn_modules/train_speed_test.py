import oneflow.experimental as flow
from oneflow.python.framework.function_util import global_function_or_identity

import numpy as np
import time
import argparse
import os
import torch

import models.pytorch_resnet50 as pytorch_resnet50
from models.resnet50 import resnet50
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image, NumpyDataLoader

def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--model_path", type=str, default="./resnet50-19c8e357.pth", help="model path"
    )
    parser.add_argument(
        "--image_path", type=str, default="./data/fish.jpg", help="input image path"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./imagenette", help="dataset path"
    )
    return parser.parse_args()

def rmse(l, r):
    return np.sqrt(np.mean(np.square(l - r)))

def main(args):
    flow.env.init()
    flow.enable_eager_execution()
    batch_size = 16
    image_nd = np.ones((batch_size, 3, 224, 224), dtype=np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)
    # train_data_loader = NumpyDataLoader(os.path.join(args.dataset_path, "train"), batch_size)
    # val_data_loader = NumpyDataLoader(os.path.join(args.dataset_path, "val"), batch_size)

    # print(len(train_data_loader), len(val_data_loader))


    # image_nd, label_nd = train_data_loader[0]
    # print(image_nd.shape, label_nd.shape)

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
            # dic[k] = torch_params[k].detach().numpy()
            torch_params[k] = torch.from_numpy(dic[k].numpy()) 

    # res50_module.load_state_dict(dic)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    # # set for eval mode
    # res50_module.eval()
    start_t = time.time()

    image = flow.Tensor(image_nd, dtype=flow.float32, requires_grad=True)
    label = flow.Tensor(label_nd, dtype=flow.int32)
    corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    image = image.to(flow.device('cuda'))
    label = label.to(flow.device('cuda'))
    res50_module.to(flow.device('cuda'))
    corss_entropy.to(flow.device('cuda'))

    learning_rate = 0.01
    mom = 0.9
    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)

    bp_iters = 10
    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    # for i in range(10):
    #     with flow.no_grad():
    #         logits = res50_module(image)
    #         loss = corss_entropy(logits, label)


    for i in range(bp_iters):
        s_t = time.time()
        logits = res50_module(image)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t
        
        s_t = time.time()
        of_sgd.step()
        of_sgd.zero_grad()
        update_time += time.time() - s_t

    of_loss = loss.numpy()
    of_in_grad = image.grad.numpy()
    predictions = logits.softmax()
    of_predictions = predictions.numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    print('fp time : {}'.format(for_time / bp_iters))
    print('bp time : {}'.format(bp_time / bp_iters))
    print('update time : {}'.format(update_time / bp_iters))
    clsidxs = np.argmax(of_predictions, axis=1)

    for i in range(batch_size):
        print("of predict prob: %f, class name: %s" % (np.max(of_predictions[i]), clsidx_2_labels[clsidxs[i]]))
    logits_of = logits.numpy()


    #####################################################################################################
    # pytorch resnet50
    torch_res50_module = pytorch_resnet50.resnet50()
    start_t = time.time()
    print(type(start_t))
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


    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    for i in range(bp_iters):
        s_t = time.time()
        logits = torch_res50_module(image)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        torch_sgd.step()
        torch_sgd.zero_grad()
        update_time += time.time() - s_t
        
    torch_loss = loss.cpu().detach().numpy()
    torch_in_grad = image.grad.cpu().detach().numpy()
    predictions = logits.softmax(-1)
    torch_predictions = predictions.cpu().detach().numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    print('fp time : {}'.format(for_time / bp_iters))
    print('bp time : {}'.format(bp_time / bp_iters))
    print('update time : {}'.format(update_time / bp_iters))
    clsidxs = np.argmax(torch_predictions, axis=1)
    for i in range(batch_size):
        print("of predict prob: %f, class name: %s" % (np.max(torch_predictions[i]), clsidx_2_labels[clsidxs[i]]))


    # check of and torch param and grad error
    print("logit rmse error: ", rmse(logits_of, logits.cpu().detach().numpy()))
    print("prediction rmse error: ", rmse(of_predictions, torch_predictions))
    print("loss rmse error: ", rmse(of_loss, torch_loss), of_loss, torch_loss)
    print("input grad rmse error: ", rmse(torch_in_grad, of_in_grad))

    # of_grads = {}
    # of_params = {}
    # for k, v in res50_module.named_parameters():
    #     of_grads[k] = v.grad.numpy() if v.grad is not None else np.random.rand(*v.numpy().shape)
    #     of_params[k] = v.numpy()

    # for k, v in torch_res50_module.named_parameters():
    #     torch_grad = v.grad.cpu().detach().numpy() if v.grad is not None else np.random.rand(*v.shape)
    #     torch_param = v.cpu().detach().numpy()

    #     if k == "fc.bias":
    #         print("of grad:", of_grads[k].flatten()[:20])
    #         print("torch grad", torch_grad.flatten()[:20])

    #         print("of param:", of_params[k].flatten()[:20])
    #         print("torch param", torch_param.flatten()[:20])


    #     if np.allclose(of_grads[k], torch_grad, atol=1e-6) == False:
    #         print("of and torch grad not match, key: %s, rmse_error: %f" % (k, rmse(of_grads[k], torch_grad)))
    #     if np.allclose(of_params[k], torch_param, atol=1e-6) == False:
    #         print("of and torch param not match, key: %s, rmse_error: %f" % (k, rmse(of_params[k], torch_param)))


if __name__ == "__main__":
    args = _parse_args()
    main(args)








