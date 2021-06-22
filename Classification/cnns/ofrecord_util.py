"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow


def add_ofrecord_args(parser):
    parser.add_argument("--image_size", type=int, default=224,
                        required=False, help="image size")
    parser.add_argument("--resize_shorter", type=int, default=256,
                        required=False, help="resize shorter for validation")
    parser.add_argument("--train_data_dir", type=str,
                        default=None, help="train dataset directory")
    parser.add_argument("--train_data_part_num", type=int,
                        default=256, help="train data part num")
    parser.add_argument("--val_data_dir", type=str,
                        default=None, help="val dataset directory")
    parser.add_argument("--val_data_part_num", type=int,
                        default=256, help="val data part num")
    return parser


def load_synthetic(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    shape=(args.image_size, args.image_size, 3) if args.channel_last else (3, args.image_size, args.image_size)
    image = flow.data.decode_random(
        shape=shape, dtype=flow.float, batch_size=batch_size
    )

    return label, image


def load_imagenet_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    output_layout="NHWC" if args.channel_last else "NCHW"

    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                        batch_size=train_batch_size,
                                        data_part_num=args.train_data_part_num,
                                        part_name_suffix_length=5,
                                        random_shuffle=True,
                                        shuffle_after_epoch=True)
    label = flow.data.OFRecordRawDecoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32)
    if args.gpu_image_decoder:
        encoded = flow.data.OFRecordBytesDecoder(ofrecord, "encoded")
        image = flow.data.ImageDecoderRandomCropResize(encoded, target_width=224, target_height=224, num_workers=3)
    else:
        image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded",  # seed=seed,
                                                        color_space=color_space)
        rsz = flow.image.Resize(image, target_size=[args.image_size, args.image_size])
        image = rsz[0]

    rng = flow.random.CoinFlip(batch_size=train_batch_size)  # , seed=seed)
    normal = flow.image.CropMirrorNormalize(image, mirror_blob=rng,
                                            color_space=color_space, output_layout=output_layout,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal


def load_imagenet_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device
    output_layout="NHWC" if args.channel_last else "NCHW"

    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                            batch_size=val_batch_size,
                                            data_part_num=args.val_data_part_num,
                                            part_name_suffix_length=5,
                                            shuffle_after_epoch=False)
    image = flow.data.OFRecordImageDecoder(
        ofrecord, "encoded", color_space=color_space)
    label = flow.data.OFRecordRawDecoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32)

    rsz = flow.image.Resize(
        image, resize_side="shorter",
        keep_aspect_ratio=True,
        target_size=args.resize_shorter)

    normal = flow.image.CropMirrorNormalize(rsz[0], color_space=color_space, output_layout=output_layout,
                                            crop_h=args.image_size, crop_w=args.image_size, crop_pos_y=0.5, crop_pos_x=0.5,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal


if __name__ == "__main__":
    import os
    import config as configs
    from util import InitNodes, Metric
    from job_function_util import get_val_config
    parser = configs.get_parser()
    args = parser.parse_args()
    configs.print_args(args)

    flow.config.gpu_device_num(args.gpu_num_per_node)
    #flow.config.enable_debug_mode(True)
    @flow.global_function(get_val_config(args))
    def IOTest():
        if args.train_data_dir:
            assert os.path.exists(args.train_data_dir)
            print("Loading data from {}".format(args.train_data_dir))
            (labels, images) = load_imagenet_for_training(args)
        else:
            print("Loading synthetic data.")
            (labels, images) = load_synthetic(args)
        outputs = {"images": images, "labels": labels}
        return outputs

    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    metric = Metric(desc='io_test', calculate_batches=args.loss_print_every_n_iter,
                    batch_size=train_batch_size, prediction_key=None)
    for i in range(1000):
        IOTest().async_get(metric.metric_cb(0, i))
