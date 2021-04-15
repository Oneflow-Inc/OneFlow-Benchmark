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
    mom = 0.9
    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)

    # # set for eval mode
    # res50_module.eval()

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, placement=flow.placement("gpu", ["0:0"], None), is_consistent=True, requires_grad=True)

    # label = flow.Tensor([1], dtype=flow.int32, requires_grad=False)
    # corss_entropy = flow.nn.CrossEntropyLoss()
    # TODO
    # loss = corss_entropy(logits, label)
    bp_iters = 5
    for i in range(bp_iters):
        logits = res50_module(image)
        grad = flow.Tensor(1, 1000)
        flow.nn.init.ones_(grad)
        grad.determine()
        @global_function_or_identity()
        def job():
            logits.backward(grad)
        job()

        # for p in res50_module.parameters():
        #     p[:] = p - learning_rate * p.grad

        of_sgd.step()
        of_sgd.zero_grad()

    of_in_grad = image.grad.numpy()
    predictions = logits.softmax()
    of_predictions = predictions.numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    clsidx = np.argmax(of_predictions)
    print("of predict prob: %f, class name: %s" % (np.max(of_predictions), clsidx_2_labels[clsidx]))
    logits_of = logits.numpy()



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
    image = load_image(args.image_path)
    image = torch.tensor(image)
    image = image.to('cuda')
    image.requires_grad = True

    for i in range(bp_iters):
        torch_sgd.zero_grad()
        logits = torch_res50_module(image)
        logits.backward(torch.ones_like(logits))
        # for p in torch_res50_module.parameters():
        #     p.data.add_(p.grad.data, alpha=-learning_rate)
        torch_sgd.step()
        

    torch_in_grad = image.grad.cpu().detach().numpy()

    predictions = logits.softmax(-1)
    torch_predictions = predictions.cpu().detach().numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    clsidx = np.argmax(torch_predictions)
    print("torch predict prob: %f, class name: %s" % (np.max(torch_predictions), clsidx_2_labels[clsidx]))



    # check of and torch param and grad error
    print(np.allclose(of_predictions, torch_predictions, atol=1e-5))
    print(np.max(np.abs(logits_of - logits.cpu().detach().numpy())))
    print(np.allclose(logits_of, logits.cpu().detach().numpy(), atol=1e-2))
    print(np.max(np.abs(torch_in_grad - of_in_grad)))
    print(np.allclose(torch_in_grad, of_in_grad, atol=1e-5))

    
    of_grads = {}
    of_params = {}
    for k, v in res50_module.named_parameters():
        of_grads[k] = v.grad.numpy()
        of_params[k] = v.numpy()

    for k, v in torch_res50_module.named_parameters():
        torch_grad = v.grad.cpu().detach().numpy()
        torch_param = v.cpu().detach().numpy() 
        if np.allclose(of_grads[k], torch_grad, atol=1e-3) == False:
            print("of and torch grad not match, key: %s, max_error: %f" % (k, np.max(np.abs(of_grads[k] - torch_grad))))
        if np.allclose(of_params[k], torch_param, atol=1e-3) == False:
            print("of and torch param not match, key: %s, max_error: %f" % (k, np.max(np.abs(of_params[k] - torch_param))))

if __name__ == "__main__":
    args = _parse_args()
    main(args)





















# train_batch_size = 1
# train_record_reader = flow.nn.OfrecordReader("./ofrecord_224/train",
#                         batch_size=train_batch_size,
#                         data_part_num=1,
#                         part_name_suffix_length=5,
#                         random_shuffle=False,
#                         shuffle_after_epoch=False)
# record_label_decoder = flow.nn.OfrecordRawDecoder("class/label", shape=(), dtype=flow.int32)
# color_space = 'RGB'
# height = 224
# width = 224
# channels = 3
# record_image_decoder = flow.nn.OfrecordRawDecoder("encoded", shape=(height, width, channels), dtype=flow.uint8)
# flip = flow.nn.CoinFlip(batch_size=train_batch_size)
# rgb_mean = [123.68, 116.779, 103.939]
# rgb_std = [58.393, 57.12, 57.375]
# crop_mirror_norm = flow.nn.CropMirrorNormalize(color_space=color_space, output_layout="NCHW",
#                                             mean=rgb_mean, std=rgb_std, output_dtype=flow.float)
# rng = flip()

# images = []
# for i in range(10):
#     start = time.time()
#     train_record = train_record_reader()
#     label = record_label_decoder(train_record)
#     image_raw_buffer = record_image_decoder(train_record)
#     image = crop_mirror_norm(image_raw_buffer, rng)
#     print(image.shape, time.time() - start)

#     # recover images
#     image_np = image.numpy()
#     images.append(image_np.copy())
#     image_np = np.squeeze(image_np)
#     image_np = np.transpose(image_np, (1, 2, 0))
#     image_np = image_np * rgb_std + rgb_mean
#     image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
#     image_np = image_np.astype(np.uint8)
#     print(image_np.shape)
#     cv2.imwrite("recover_image%d.jpg" % i, image_np)

# for i in range(1, 10):
#     print(np.allclose(images[0], images[i]))




