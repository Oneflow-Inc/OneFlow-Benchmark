from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape,
                 min_random_area, max_random_area,
                 min_random_aspect_ratio, max_random_aspect_ratio,
                 nvjpeg_padding, hw_decoder_load, dont_use_mmap, prefetch_queue=3,
                 seed=12,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True, use_roi_decode=False, cache_size=0):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id,
            seed=seed + device_id,
            prefetch_queue_depth=prefetch_queue)

        _mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
        _std_pixel = [255 * x for x in (0.229, 0.224, 0.225)]

        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     lazy_init=True, dont_use_mmap=dont_use_mmap,
                                     random_shuffle=True, shard_id=shard_id, num_shards=num_shards)
        self.decode = ops.ImageDecoderRandomCrop(device="mixed", output_type=types.RGB,
                                                 device_memory_padding=nvjpeg_padding,
                                                 host_memory_padding=nvjpeg_padding,
                                                 random_area=[
                                                     min_random_area,
                                                     max_random_area],
                                                 random_aspect_ratio=[
                                                     min_random_aspect_ratio,
                                                     max_random_aspect_ratio],
                                                 affine=False)
        self.rrc = ops.Resize(device="gpu", resize_x=crop_shape[0],
                              resize_y=crop_shape[1])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout,
                                            crop=crop_shape,
                                            pad_output=pad_output,
                                            image_type=types.RGB,
                                            mean=_mean_pixel,
                                            std=_std_pixel)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape,
                 nvjpeg_padding, hw_decoder_load, dont_use_mmap, prefetch_queue=3,
                 seed=12, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True, cache_size=0):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id,
            seed=seed + device_id,
            prefetch_queue_depth=prefetch_queue)

        _mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
        _std_pixel = [255 * x for x in (0.229, 0.224, 0.225)]

        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     dont_use_mmap=dont_use_mmap,
                                     lazy_init=True,
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                       hw_decoder_load=hw_decoder_load,
                                       device_memory_padding=nvjpeg_padding,
                                       host_memory_padding=nvjpeg_padding,
                                       affine=False)

        self.resize = ops.Resize(device="gpu", resize_shorter=resize_shp) if resize_shp else None

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout=output_layout,
                                            crop=crop_shape,
                                            pad_output=pad_output,
                                            image_type=types.RGB,
                                            mean=_mean_pixel,
                                            std=_std_pixel)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        return [output, self.labels]


def build_input_pipeline():
    # resize is default base length of shorter edge for dataset;
    # all images will be reshaped to this size
    resize = 256
    # target shape is final shape of images pipelined to network;
    # all images will be cropped to this size
    target_shape = [4, 224, 224]
    pad_output = target_shape[0] == 4

    trainpipe = HybridTrainPipe(batch_size=32,
                                num_threads=3,
                                device_id=0,
                                rec_path="/dataset/imagenet-mxnet/train.rec",
                                idx_path="/dataset/imagenet-mxnet/train.idx",
                                shard_id=0,
                                num_shards=1,
                                crop_shape=target_shape[1:],
                                min_random_area=0.05,
                                max_random_area=1.0,
                                min_random_aspect_ratio=0.75,
                                max_random_aspect_ratio=1.3333333333333333,
                                nvjpeg_padding=256 * 1024 * 1024,
                                prefetch_queue=3,
                                hw_decoder_load=0.0,
                                dont_use_mmap=False,
                                seed=41905,
                                output_layout=types.NHWC,
                                pad_output=pad_output,
                                dtype='float16',
                                mlperf_print=False,
                                use_roi_decode=True,
                                cache_size=0)

    inferencepipe = HybridValPipe(batch_size=32,
                                  num_threads=3,
                                  device_id=0,
                                  rec_path="/dataset/imagenet-mxnet/val.rec",
                                  idx_path="/dataset/imagenet-mxnet/val.idx",
                                  shard_id=0,
                                  num_shards=1,
                                  crop_shape=target_shape[1:],
                                  nvjpeg_padding=256 * 1024 * 1024,
                                  prefetch_queue=3,
                                  seed=41905,
                                  dont_use_mmap=False,
                                  hw_decoder_load=0.0,
                                  resize_shp=resize,
                                  output_layout=types.NHWC,
                                  pad_output=pad_output,
                                  dtype='float16',
                                  mlperf_print=False,
                                  cache_size=0)

    trainpipe.build()
    inferencepipe.build()

    train_iter = DALIClassificationIterator(trainpipe, 1281167)
    inference_iter = DALIClassificationIterator(inferencepipe, 50000, fill_last_batch=False)

    return train_iter, inference_iter
