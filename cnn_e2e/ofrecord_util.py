from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow

def add_ofrecord_args(parser):
    parser.add_argument("--image_size", type=int, default=224, required=False, help="image size")
    parser.add_argument("--train_data_dir", type=str, default=None, help="train dataset directory")
    parser.add_argument("--train_data_part_num", type=int, default=256, help="train data part num")
    parser.add_argument("--val_data_dir", type=str, default=None, help="val dataset directory")
    parser.add_argument("--val_data_part_num", type=int, default=256, help="val data part num")
    return parser


def load_imagenet(args, batch_size, data_dir, data_part_num, codec):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(args.image_size, args.image_size, 3),
        dtype=flow.float,
        codec=codec,
        preprocessors=[flow.data.NormByChannelPreprocessor(args.rgb_mean, args.rgb_std)],
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=batch_size,
        data_part_num=data_part_num,
        part_name_suffix_length=5,
        shuffle = True,
        buffer_size=32768,
        name="decode",
    )


def load_imagenet_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    codec=flow.data.ImageCodec([
        flow.data.ImageCropWithRandomSizePreprocessor(area=(0.08, 1)),
        flow.data.ImageResizePreprocessor(args.image_size, args.image_size),
        flow.data.ImagePreprocessor('mirror'),
    ])
    return load_imagenet(args, train_batch_size, args.train_data_dir, args.train_data_part_num,
                         codec)


def load_imagenet_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device
    codec=flow.data.ImageCodec(
        [
            flow.data.ImageTargetResizePreprocessor(resize_shorter=256),
            flow.data.ImageCenterCropPreprocessor(args.image_size, args.image_size),
            #flow.data.ImageResizePreprocessor(args.image_size, args.image_size),
        ]
    )
    return load_imagenet(args, val_batch_size, args.val_data_dir, args.val_data_part_num, codec)


def load_synthetic(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    image = flow.data.decode_random(
        shape=(args.image_size, args.image_size, 3), dtype=flow.float, batch_size=batch_size
    )

    return label, image
