import oneflow.experimental as flow

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
        "--dataset_path", type=str, default="./ofrecord", help="dataset path"
    )
    return parser.parse_args()

def rmse(l, r):
    return np.sqrt(np.mean(np.square(l - r)))

def main(args):
    flow.env.init()
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    #############################################
    train_batch_size = 16
    val_batch_size = 16
    channel_last = False
    output_layout = "NHWC" if channel_last else "NCHW"
    train_record_reader = flow.nn.OfrecordReader(os.path.join(args.dataset_path, "train"),
                            batch_size=train_batch_size,
                            data_part_num=1,
                            part_name_suffix_length=5,
                            random_shuffle=True,
                            shuffle_after_epoch=True)
    record_label_decoder = flow.nn.OfrecordRawDecoder("class/label", shape=(), dtype=flow.int32)
    color_space = 'RGB'
    height = 224
    width = 224
    channels = 3
    record_image_decoder = flow.nn.OFRecordImageDecoderRandomCrop("encoded", color_space=color_space)
    resize = flow.nn.image.Resize(target_size=[height, width])

    flip = flow.nn.CoinFlip(batch_size=train_batch_size)
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    crop_mirror_norm = flow.nn.CropMirrorNormalize(color_space=color_space, output_layout=output_layout,
                                                mean=rgb_mean, std=rgb_std, output_dtype=flow.float)

    val_record_reader = flow.nn.OfrecordReader(os.path.join(args.dataset_path, "val"),
                                            batch_size=val_batch_size,
                                            data_part_num=1,
                                            part_name_suffix_length=5,
                                            shuffle_after_epoch=False)
    val_record_image_decoder = flow.nn.OFRecordImageDecoder("encoded", color_space=color_space)
    val_resize = flow.nn.image.Resize(resize_side="shorter", keep_aspect_ratio=True, target_size=256)
    val_crop_mirror_normal = flow.nn.CropMirrorNormalize(color_space=color_space, output_layout=output_layout,
                                                        crop_h=height, crop_w=width, crop_pos_y=0.5, crop_pos_x=0.5,
                                                        mean=rgb_mean, std=rgb_std, output_dtype=flow.float)

    train_set_size = 9469
    val_set_size = 3925
    train_loop = train_set_size // train_batch_size
    val_loop = val_set_size // val_batch_size
    ###################################################

    epochs = 1000
    learning_rate = 0.001
    mom = 0.9

    ###############################
    # pytorch init
    torch_res50_module = pytorch_resnet50.resnet50()
    start_t = time.time()
    print(type(start_t))
    torch_params = torch_res50_module.state_dict()
    end_t = time.time()
    print('torch load params time : {}'.format(end_t - start_t))
    torch_res50_module.to('cuda')
    torch_sgd = torch.optim.SGD(torch_res50_module.parameters(), lr=learning_rate, momentum=mom)
    
    corss_entropy = torch.nn.CrossEntropyLoss()
    corss_entropy.to('cuda')
    ###############################

    #################
    # oneflow init
    start_t = time.time()
    res50_module = resnet50()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))

    # flow.save(res50_module.state_dict(), "./save_model")

    start_t = time.time()
    torch_keys = torch_params.keys()

    dic = res50_module.state_dict()
    for k in dic.keys():
        if k in torch_keys:
            dic[k] = torch_params[k].numpy()
    res50_module.load_state_dict(dic)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    of_corss_entropy = flow.nn.CrossEntropyLoss()

    res50_module.to(flow.device('cuda'))
    of_corss_entropy.to(flow.device('cuda'))

    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)


    ############################
    of_losses = []
    torch_losses = []

    all_samples = val_loop * val_batch_size

    for epoch in range(epochs):
        res50_module.train()
        torch_res50_module.train()

        for b in range(train_loop):
        # for b in range(100):
            print("epoch %d train iter %d" % (epoch, b))
            with flow.no_grad():
                train_record = train_record_reader()
                label = record_label_decoder(train_record)
                image_raw_buffer = record_image_decoder(train_record)
                image = resize(image_raw_buffer)
                rng = flip()
                image = crop_mirror_norm(image, rng)
        
            # oneflow train 
            start_t = time.time()
            image = image.to(flow.device('cuda'))
            label = label.to(flow.device('cuda'))
            logits = res50_module(image)
            loss = of_corss_entropy(logits, label)
            loss.backward()
            of_sgd.step()
            of_sgd.zero_grad()
            end_t = time.time()
            l = loss.numpy()[0]
            of_losses.append(l)
            print('oneflow loss {}, train time : {}'.format(l, end_t - start_t))

            # pytroch train
            start_t = time.time()
            image = torch.from_numpy(image.numpy()).to('cuda')
            label = torch.tensor(label.numpy(), dtype=torch.long, requires_grad=False).to('cuda')
            logits = torch_res50_module(image)
            loss = corss_entropy(logits, label)
            loss.backward()
            torch_sgd.step()
            torch_sgd.zero_grad()
            end_t = time.time()
            l = loss.cpu().detach().numpy()
            torch_losses.append(l)
            print('pytorch loss {}, train time : {}'.format(l, end_t - start_t))
        
        print("epoch %d done, start validation" % epoch)

        res50_module.eval()
        torch_res50_module.eval()
        correct_of = 0.0
        correct_torch = 0.0
        for b in range(val_loop):
        # for b in range(0):
            print("epoch %d val iter %d" % (epoch, b))
            with flow.no_grad():
                val_record = val_record_reader()
                label = record_label_decoder(val_record)
                image_raw_buffer = val_record_image_decoder(val_record)
                image = val_resize(image_raw_buffer)
                image = val_crop_mirror_normal(image)

            start_t = time.time()
            image = image.to(flow.device('cuda'))
            with flow.no_grad():
                logits = res50_module(image)
                predictions = logits.softmax()
            of_predictions = predictions.numpy()
            clsidxs = np.argmax(of_predictions, axis=1)

            label_nd = label.numpy()
            for i in range(val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()
            print("of predict time: %f, %d" % (end_t - start_t, correct_of))

            # pytroch val
            start_t = time.time()
            image = torch.from_numpy(image.numpy()).to('cuda')
            with torch.no_grad():
                logits = torch_res50_module(image)
                predictions = logits.softmax(-1)
            torch_predictions = predictions.cpu().detach().numpy()
            clsidxs = np.argmax(torch_predictions, axis=1)
            for i in range(val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_torch += 1
            end_t = time.time()
            print("torch predict time: %f, %d" % (end_t - start_t, correct_torch))

        print("epoch %d, oneflow top1 val acc: %f, torch top1 val acc: %f" % (epoch, correct_of / all_samples, correct_torch / all_samples))

    writer = open("of_losses.txt", "w")
    for o in of_losses:
        writer.write("%f\n" % o)
    writer.close()

    writer = open("torch_losses.txt", "w")
    for o in torch_losses:
        writer.write("%f\n" % o)
    writer.close()



if __name__ == "__main__":
    args = _parse_args()
    main(args)








