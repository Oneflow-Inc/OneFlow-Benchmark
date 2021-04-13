import oneflow as flow
import oneflow_api
# from oneflow.python.oneflow_export import oneflow_export
# from oneflow.python.nn.module import Module
# from oneflow.python.nn.modules.utils import (
#     _single,
#     _pair,
#     _triple,
#     _reverse_repeat_tuple,
# )
# from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
# from typing import Optional, List, Tuple, Sequence

import numpy as np
import cv2
import time


flow.env.init()
flow.InitEagerGlobalSession()
flow.enable_eager_execution(True)


train_batch_size = 1
train_record_reader = flow.nn.OfrecordReader("./ofrecord_224/val",
                        batch_size=train_batch_size,
                        data_part_num=1,
                        part_name_suffix_length=5,
                        random_shuffle=False,
                        shuffle_after_epoch=False)
record_label_decoder = flow.nn.OfrecordRawDecoder("class/label", shape=(), dtype=flow.int32)
color_space = 'RGB'
height = 224
width = 224
channels = 3
record_image_decoder = flow.nn.OfrecordRawDecoder("encoded", shape=(height, width, channels), dtype=flow.uint8)
flip = flow.nn.CoinFlip(batch_size=train_batch_size)
rgb_mean = [123.68, 116.779, 103.939]
rgb_std = [58.393, 57.12, 57.375]
crop_mirror_norm = flow.nn.CropMirrorNormalize(color_space=color_space, output_layout="NCHW",
                                            mean=rgb_mean, std=rgb_std, output_dtype=flow.float)
rng = flip()


for i in range(10):
    start = time.time()
    train_record = train_record_reader()
    label = record_label_decoder(train_record)
    image_raw_buffer = record_image_decoder(train_record)
    image = crop_mirror_norm(image_raw_buffer, rng)
    print(image.shape, time.time() - start)

    # recover images
    image_np = image.numpy()
    image_np = np.squeeze(image_np)
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = image_np * rgb_std + rgb_mean
    image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
    image_np = image_np.astype(np.uint8)
    print(image_np.shape)
    cv2.imwrite("recover_image%d.jpg" % i, image_np)
