# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import time
import ctypes
#import logging
import warnings
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.backend import TensorGPU


def add_dali_args(parser):
    group = parser.add_argument_group('DALI data backend',
                                      'entire group applies only to dali data backend')
    group.add_argument('--dali-separ-val', action='store_true',
                       help='each process will perform independent validation on whole val-set')
    group.add_argument('--dali-threads', type=int, default=3, help="number of threads" +\
                       "per GPU for DALI")
    group.add_argument('--dali-validation-threads', type=int, default=10,
                       help="number of threads per GPU for DALI for validation")
    group.add_argument('--dali-prefetch-queue', type=int, default=2,
                       help="DALI prefetch queue depth")
    group.add_argument('--dali-nvjpeg-memory-padding', type=int, default=64,
                       help="Memory padding value for nvJPEG (in MB)")
    group.add_argument('--dali-fuse-decoder', type=int, default=1,
                       help="0 or 1 whether to fuse decoder or not")
    return parser


class HybridTrainPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id,
                                              seed=12 + device_id,
                                              prefetch_queue_depth = prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=True, shard_id=shard_id, num_shards=num_shards)
        self.cast = ops.Cast(dtype=types.INT32)

        dali_device = "cpu" if dali_cpu else "mixed"
        dali_resize_device = "cpu" if dali_cpu else "gpu"

        if args.dali_fuse_decoder:
            self.decode = ops.ImageDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                     device_memory_padding=nvjpeg_padding,
                                                     host_memory_padding=nvjpeg_padding)
            self.resize = ops.Resize(device=dali_resize_device, resize_x=crop_shape[1],
                                     resize_y=crop_shape[0])
        else:
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB,
                                           device_memory_padding=nvjpeg_padding,
                                           host_memory_padding=nvjpeg_padding)
            self.resize = ops.RandomResizedCrop(device=dali_resize_device, size=crop_shape)


        self.cmnp = ops.CropMirrorNormalize(device=dali_resize_device,
            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
            image_type=types.RGB, mean=args.rgb_mean, std=args.rgb_std)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.cast(self.labels)]


class HybridValPipe(Pipeline):
    def __init__(self, args, batch_size, num_threads, device_id, rec_path, idx_path, shard_id,
                 num_shards, crop_shape, nvjpeg_padding, prefetch_queue=3, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16', dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,
                                            prefetch_queue_depth=prefetch_queue)
        self.input = ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        self.cast = ops.Cast(dtype=types.INT32)
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB,
                                           device_memory_padding=nvjpeg_padding,
                                           host_memory_padding=nvjpeg_padding)
        self.resize = ops.Resize(device=dali_device, resize_shorter=resize_shp) if resize_shp else None
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
            output_dtype=types.FLOAT16 if dtype == 'float16' else types.FLOAT,
            output_layout=output_layout, crop=crop_shape, pad_output=pad_output,
            image_type=types.RGB, mean=args.rgb_mean, std=args.rgb_std)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        #return [output, self.labels]
        return [output, self.cast(self.labels)]


def feed_ndarray(dali_tensor, arr):
    """
    Copy contents of DALI tensor to numpy's NDArray.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : numpy.NDArray
            Destination of the copy
    """
    # Wait until arr is no longer used by the engine
    assert dali_tensor.shape() == list(arr.shape), \
            ("Shapes do not match: DALI tensor has shape {0}"
            ", but NDArray has shape {1}".format(dali_tensor.shape(), list(arr.shape)))
    # Get CTypes void pointer to the underlying memory held by arr
    c_type_pointer = ctypes.c_void_p(arr.ctypes.data)
    # Copy data from DALI tensor to ptr
    dali_tensor.copy_to_external(c_type_pointer)


class DALIGenericIterator(object):
    """
    General DALI iterator for Numpy. It can return any number of
    outputs from the DALI pipeline in the form of ndarray.

    Parameters
    ----------
    pipelines : list of nvidia.dali.pipeline.Pipeline
                List of pipelines to use
    size : int, Number of samples in the epoch (Usually the size of the dataset).
    data_layout : str, optional, default = 'NCHW'
                  Either 'NHWC' or 'NCHW' - layout of the pipeline outputs.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    squeeze_labels: bool, optional, default = True
                 Whether the iterator should squeeze the labels before
                 copying them to the ndarray.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the DALI pipeline can
                 change during execution. If True, the ndarray will be resized accordingly
                 if the shape of DALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by DALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one. For this happen, the option `pad_last_batch`
                 in the reader need to be set to `True` as well.

    Example
    -------
    With the data set [1,2,3,4,5,6,7] and the batch size 2:
    fill_last_batch = False, last_batch_padded = True  -> last batch = [7], next iteration will return [1, 2]
    fill_last_batch = False, last_batch_padded = False -> last batch = [7], next iteration will return [2, 3]
    fill_last_batch = True, last_batch_padded = True   -> last batch = [7, 7], next iteration will return [1, 2]
    fill_last_batch = True, last_batch_padded = False  -> last batch = [7, 1], next iteration will return [2, 3]
    """
    def __init__(self,
                 pipelines,
                 size,
                 output_map=['data', 'label'],
                 data_layout='NCHW',
                 fill_last_batch=False,
                 auto_reset=False,
                 squeeze_labels=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self._num_gpus = len(pipelines)
        assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        self.batch_size = pipelines[0].batch_size
        self._size = int(size)
        self._pipes = pipelines
        self._fill_last_batch = fill_last_batch
        self._last_batch_padded = last_batch_padded
        self._auto_reset = auto_reset
        self._squeeze_labels = squeeze_labels
        self._dynamic_shape = dynamic_shape
        # Build all pipelines
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.build()
        # Use double-buffering of data batches
        self._data_batches = [[None] for i in range(self._num_gpus)]
        self._counter = 0
        self._current_data_batch = 0
        self.output_map = output_map

        # We need data about the batches (like shape information),
        # so we need to run a single batch as part of setup to get that info
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.schedule_run()
        self._first_batch = None
        self._first_batch = self.next()


    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        if self._counter >= self._size:
            if self._auto_reset:
                self.reset()
            raise StopIteration
        # Gather outputs
        outputs = []
        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                outputs.append(p.share_outputs())
        for i in range(self._num_gpus):
            # MXNet wants batches with clear distinction between
            # data and label entries, so segregate outputs into
            # 2 categories
            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_info = dict()
            for j, out in enumerate(outputs[i]):
                x = out.as_tensor()
                category_tensors[self.output_map[j]] = x#.as_tensor()
                if self._squeeze_labels and self.output_map[j]=='label':
                    category_tensors[self.output_map[j]].squeeze()
                category_info[self.output_map[j]] = (x.shape(), np.dtype(x.dtype()))

            # If we did not yet allocate memory for that batch, do it now
            if self._data_batches[i][self._current_data_batch] is None:
                for category in self.output_map:
                    t = category_tensors[category]
                    assert type(t) is not TensorGPU, "CPU data only"#TODO
                d = []
                for (shape, dtype) in category_info.values():
                    d.append(np.zeros(shape, dtype = dtype))

                self._data_batches[i][self._current_data_batch] = d

            d = self._data_batches[i][self._current_data_batch]
            # Copy data from DALI Tensors to NDArrays
            if self._dynamic_shape:
                for j, (shape, dtype) in enumerate(category_info):
                    if list(d[j].shape) != shape:
                        d[j] = np.zeros(shape, dtype = dtype)

            for j, d_arr in enumerate(d):
                feed_ndarray(category_tensors[self.output_map[j]], d_arr)

        for p in self._pipes:
            with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                p.release_outputs()
                p.schedule_run()

        copy_db_index = self._current_data_batch
        # Change index for double buffering
        self._current_data_batch = (self._current_data_batch + 1) % 1
        self._counter += self._num_gpus * self.batch_size

        assert not self._fill_last_batch
        ## padding the last batch
        #if (not self._fill_last_batch) and (self._counter > self._size):
        #        # this is the last batch and we need to pad
        #        overflow = self._counter - self._size
        #        overflow_per_device = overflow // self._num_gpus
        #        difference = self._num_gpus - (overflow % self._num_gpus)
        #        for i in range(self._num_gpus):
        #            if i < difference:
        #                self._data_batches[i][copy_db_index].pad = overflow_per_device
        #            else:
        #                self._data_batches[i][copy_db_index].pad = overflow_per_device + 1
        #else:
        #    for db in self._data_batches:
        #        db[copy_db_index].pad = 0

        #_data_batches[gpu_id][_current_data_batch][images, labels]
        images = [db[copy_db_index][0] for db in self._data_batches]
        labels = [db[copy_db_index][1] for db in self._data_batches]
        #return images, labels
        return np.concatenate(images), np.concatenate(labels)
        #return [db[copy_db_index] for db in self._data_batches]

    def next(self):
        """
        Returns the next batch of data.
        """
        return self.__next__()

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        if self._counter >= self._size:
            if self._fill_last_batch and not self._last_batch_padded:
                self._counter = self._counter % self._size
            else:
                self._counter = 0
            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            print("DALI iterator does not support resetting while epoch is not finished. Ignoring...")


def get_rec_iter(args, dali_cpu=False, concat=True):
    gpus = range(args.gpu_num_per_node)
    rank = 0 #TODO
    nWrk = 1 #TODO

    num_threads = args.dali_threads
    num_validation_threads = args.dali_validation_threads
    pad_output = (args.image_shape[0] == 4)

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    trainpipes = [HybridTrainPipe(args           = args,
                                  batch_size     = args.batch_size_per_device,
                                  num_threads    = num_threads,
                                  device_id      = gpu_id,
                                  rec_path       = args.data_train,
                                  idx_path       = args.data_train_idx,
                                  shard_id       = gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards     = len(gpus)*nWrk,
                                  crop_shape     = args.image_shape[1:],
                                  output_layout  = output_layout,
                                  dtype          = args.dtype,
                                  pad_output     = pad_output,
                                  dali_cpu       = dali_cpu,
                                  nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus]

    if args.data_val:
        valpipes = [HybridValPipe(args           = args,
                                  batch_size     = args.val_batch_size_per_device,
                                  num_threads    = num_validation_threads,
                                  device_id      = gpu_id,
                                  rec_path       = args.data_val,
                                  idx_path       = args.data_val_idx,
                                  shard_id       = gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards     = len(gpus)*nWrk,
                                  crop_shape     = args.image_shape[1:],
                                  resize_shp     = 256, #args.data_val_resize,
                                  output_layout  = output_layout,
                                  dtype          = args.dtype,
                                  pad_output     = pad_output,
                                  dali_cpu       = dali_cpu,
                                  nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue = args.dali_prefetch_queue) for gpu_id in gpus]
    trainpipes[0].build()
    if args.data_val:
        valpipes[0].build()
        val_examples = valpipes[0].epoch_size("Reader")

    if args.num_examples < trainpipes[0].epoch_size("Reader"):
        warnings.warn("{} training examples will be used, although full training set contains {} examples".format(args.num_examples, trainpipes[0].epoch_size("Reader")))

    dali_train_iter = DALIGenericIterator(trainpipes, args.num_examples)
    if args.data_val:
        dali_val_iter = DALIGenericIterator(valpipes, val_examples, fill_last_batch = False)
    else:
        dali_val_iter = None

    return dali_train_iter, dali_val_iter


if __name__ == '__main__':
    import config as configs
    parser = configs.get_parser()
    args = parser.parse_args()
    configs.print_args(args)
    train_data_iter, val_data_iter = get_rec_iter(args, True)
    for epoch in range(args.num_epochs):
        tic = time.time()
        last_time = time.time()
        print('Starting epoch {}'.format(epoch))
        train_data_iter.reset()
        for i, batches in enumerate(train_data_iter):
            images, labels = batches
            if i % args.loss_print_every_n_iter == 0:
                print(args.loss_print_every_n_iter * 256 / (time.time() - last_time))
                last_time = time.time()
            #print(images.shape)
        epoch_time = time.time() - tic
        print('epoch mena images/sec', 1281167 / epoch_time, epoch_time)
