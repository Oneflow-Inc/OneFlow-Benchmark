from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow

def add_ofrecord_args(parser):
    parser.add_argument("--image_size", type=int, default=224, required=False, help="image size")
    parser.add_argument("--resize_shorter", type=int, default=256, required=False, help="resize shorter for validation")
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
        preprocessors=[flow.data.NormByChannelPreprocessor(args.rgb_mean[::-1],
                                                           args.rgb_std[::-1])],
        #preprocessors=[flow.data.NormByChannelPreprocessor(args.rgb_mean, args.rgb_std)], #bgr2rgb
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
        #shuffle = True,
        #buffer_size=32768,
        name="decode",
    )


def load_imagenet_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    codec=flow.data.ImageCodec([
        #flow.data.ImagePreprocessor('bgr2rgb'),
        #flow.data.ImageCropWithRandomSizePreprocessor(area=(0.08, 1)),
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
            #flow.data.ImagePreprocessor('bgr2rgb'),
            # flow.data.ImageTargetResizePreprocessor(resize_shorter=256),
            # flow.data.ImageCenterCropPreprocessor(args.image_size, args.image_size),
            flow.data.ImageResizePreprocessor(args.image_size, args.image_size),
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


def load_imagenet_for_training2(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device

    color_space = 'RGB'
    with flow.fixed_placement("cpu", "0:0-{}".format(args.gpu_num_per_node - 1)):
        ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                             batch_size=train_batch_size,
                                             data_part_num=args.train_data_part_num,
                                             part_name_suffix_length=5,
                                             random_shuffle = True,
                                             shuffle_after_epoch=True)
        image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded", #seed=seed,
                                                         color_space=color_space)
        label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
        rsz = flow.image.Resize(image, resize_x=args.image_size, resize_y=args.image_size,
                                color_space=color_space)

        rng = flow.random.CoinFlip(batch_size=train_batch_size)#, seed=seed)
        normal = flow.image.CropMirrorNormalize(rsz, mirror_blob=rng, color_space=color_space,
            mean=args.rgb_mean, std=args.rgb_std, output_dtype = flow.float)
        return label, normal

def load_imagenet_for_validation2(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device

    color_space = 'RGB'
    with flow.fixed_placement("cpu", "0:0-{}".format(args.gpu_num_per_node - 1)):
        ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                             batch_size=val_batch_size,
                                             data_part_num=args.val_data_part_num,
                                             part_name_suffix_length=5,
                                             shuffle_after_epoch=False)
        image = flow.data.OFRecordImageDecoder(ofrecord, "encoded", color_space=color_space)
        label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
        rsz = flow.image.Resize(image, resize_shorter=args.resize_shorter, color_space=color_space)

        normal = flow.image.CropMirrorNormalize(rsz, color_space=color_space,
            crop_h = args.image_size, crop_w = args.image_size, crop_pos_y = 0.5, crop_pos_x = 0.5,
            mean=args.rgb_mean, std=args.rgb_std, output_dtype = flow.float)
        return label, normal

if __name__ == "__main__":
    import os
    import config as configs
    from util import Summary, InitNodes, Metric
    from job_function_util import get_val_config
    parser = configs.get_parser()
    args = parser.parse_args()
    configs.print_args(args)

    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.enable_debug_mode(True)
    @flow.function(get_val_config(args))
    def IOTest():
        if args.train_data_dir:
            assert os.path.exists(args.train_data_dir)
            print("Loading data from {}".format(args.train_data_dir))
            (labels, images) = load_imagenet_for_training(args)
            #(labels, images) = load_imagenet_for_training2(args)
        else:
            print("Loading synthetic data.")
            (labels, images) = load_synthetic(args)
        outputs = {"images":images, "labels": labels}
        return outputs

    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    summary = Summary(args.log_dir, args, filename='io_test.csv')
    metric = Metric(desc='io_test', calculate_batches=args.loss_print_every_n_iter,
                    summary=summary, save_summary_steps=args.loss_print_every_n_iter,
                    batch_size=train_batch_size, prediction_key=None)
    for i in range(1000):
        IOTest().async_get(metric.metric_cb(0, i))
