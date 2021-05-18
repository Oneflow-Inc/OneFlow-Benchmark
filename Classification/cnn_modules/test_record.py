import oneflow.experimental as flow
import numpy as np
import cv2
import time

flow.env.init()
flow.enable_eager_execution()
flow.InitEagerGlobalSession()


channel_last = False
output_layout = "NHWC" if channel_last else "NCHW"
batch_size = 16
train_record_reader = flow.nn.OfrecordReader("/home/ldpe2g/oneFlow/oneflowBenckmark/dataset/imagenette/ofrecord/train",
                        batch_size=batch_size,
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

flip = flow.nn.CoinFlip(batch_size=batch_size)
rgb_mean = [123.68, 116.779, 103.939]
rgb_std = [58.393, 57.12, 57.375]
crop_mirror_norm = flow.nn.CropMirrorNormalize(color_space=color_space, output_layout=output_layout,
                                            mean=rgb_mean, std=rgb_std, output_dtype=flow.float)

with flow.no_grad():
    rng = flip()

# val_record_reader = flow.nn.OfrecordReader("/home/ldpe2g/oneFlow/oneflowBenckmark/dataset/imagenette/ofrecord/val",
#                                           batch_size=batch_size,
#                                           data_part_num=1,
#                                           part_name_suffix_length=5,
#                                           shuffle_after_epoch=False)
# val_record_image_decoder = flow.nn.OFRecordImageDecoder("encoded", color_space=color_space)
# val_resize = flow.nn.image.Resize(resize_side="shorter", keep_aspect_ratio=True, target_size=256)
# val_crop_mirror_normal = flow.nn.CropMirrorNormalize(color_space=color_space, output_layout=output_layout,
#                                                     crop_h=height, crop_w=width, crop_pos_y=0.5, crop_pos_x=0.5,
#                                                     mean=rgb_mean, std=rgb_std, output_dtype=flow.float)

train_set_size = 9469
val_set_size = 3925
train_loop = train_set_size // batch_size
val_loop = val_set_size // batch_size

# train
for i in range(1000000000000000000000):
    print(i)
    start = time.time()
    with flow.no_grad():
        train_record = train_record_reader()
        label = record_label_decoder(train_record)
        image_raw_buffer = record_image_decoder(train_record)
        # image = resize(image_raw_buffer)
        # image = crop_mirror_norm(image, rng)
    # print(image.shape, time.time() - start)

    # recover images
    # image_np = image.numpy()
    # image_np = np.squeeze(image_np)
    # image_np = np.transpose(image_np, (1, 2, 0))
    # image_np = image_np * rgb_std + rgb_mean
    # image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
    # image_np = image_np.astype(np.uint8)
    # print(image_np.shape)
    # cv2.imwrite("recover_image%d.jpg" % i, image_np)

# validation
# for i in range(val_loop):
#     print(i)
#     start = time.time()
#     with flow.no_grad():
#         val_record = val_record_reader()
#         label = record_label_decoder(val_record)
#         image_raw_buffer = val_record_image_decoder(val_record)
#         image = val_resize(image_raw_buffer)
#         image = val_crop_mirror_normal(image)
#     print(image.shape, time.time() - start)

    # # recover images
    # image_np = image.numpy()
    # image_np = np.squeeze(image_np)
    # image_np = np.transpose(image_np, (1, 2, 0))
    # image_np = image_np * rgb_std + rgb_mean
    # image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
    # image_np = image_np.astype(np.uint8)
    # print(image_np.shape)
    # cv2.imwrite("recover_val_image%d.jpg" % i, image_np)
