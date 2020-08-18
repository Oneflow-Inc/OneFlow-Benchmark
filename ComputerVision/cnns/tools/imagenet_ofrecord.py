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
# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Converts ImageNet data to OFRecords file format with Example protos.

The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
  data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
  ...

where 'n01440764' is the unique synset label associated with
these images.

The training data set consists of 1000 sub-directories (i.e. labels)
each containing 1200 JPEG images for a total of 1.2M JPEG images.

The evaluation data set consists of 1000 sub-directories (i.e. labels)
each containing 50 JPEG images for a total of 50K JPEG images.

  train_directory/part-00000
  train_directory/part-00001
  ...
  train_directory/part-01023

and

  validation_directory/part-00000
  validation_directory/part-00001
  ...
  validation_directory/part-01023

Each record within the OFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""

from datetime import datetime
import os
import random
import sys
import threading
import re 
import argparse
import glob

import numpy as np
import six
import cv2
import oneflow.core.record.record_pb2 as of_record
import struct

"""
# train dataset to ofrecord
python3 imagenet_ofrecord.py \
--train_directory data/imagenet/train  \
--output_directory data/imagenet/ofrecord/train   \
--label_file imagenet_lsvrc_2015_synsets.txt   \
--shards 1024  --num_threads 8 --name train  \
--bounding_box_file imagenet_2012_bounding_boxes.csv   \
--height 224 --width 224

# val dataset to ofrecord
python3 imagenet_ofrecord.py \
--validation_directory data/imagenet/validation  \
--output_directory data/imagenet/ofrecord/validation  \
--label_file imagenet_lsvrc_2015_synsets.txt --name validation  \
--shards 32 --num_threads 4 --name validation \
--bounding_box_file imagenet_2012_bounding_boxes.csv  \
--height 224 --width 224
"""


arg_parser = argparse.ArgumentParser(description = 
            'The python script to resize pics ')

arg_parser.add_argument('--resize', dest = 'resize', default = False, help = 'resize image')

arg_parser.add_argument('--name', dest='name', default='train', \
help = 'data_file_type')

arg_parser.add_argument('--width', dest='width', default=0, \
type=int, help='fixed image width')
arg_parser.add_argument('--height', dest='height', default=0, \
type=int, help='fixed image height')

arg_parser.add_argument('--train_directory', dest = 'train_directory',\
default='/tmp/', help='Training data directory')
arg_parser.add_argument('--validation_directory', dest = 'validation_directory', \
default='/tmp/', help='Validation data directory')
arg_parser.add_argument('--output_directory', dest = 'output_directory', \
default='/tmp/', help = 'Output data directory')

arg_parser.add_argument('--shards', dest='shards',\
default=1024, type=int, help='Number of shards in making OFRecord files.')


arg_parser.add_argument('--num_threads', dest='num_threads', default = 8, \
type=int, help='Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
arg_parser.add_argument('--label_file', dest = 'labels_file', \
default = 'imagenet_lsvrc_2015_synsets.txt', help = 'Labels file')

# This file containing mapping from synset to human-readable label.
# Assumes each line of the file looks like:
#
#   n02119247    black fox
#   n02119359    silver fox
#   n02119477    red fox, Vulpes fulva
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <synset>\t<human readable label>.
arg_parser.add_argument('--imagenet_metadata_file', dest = 'imagenet_metadata_file', \
default = 'imagenet_metadata.txt', help = 'ImageNet metadata file')

# This file is the output of process_bounding_box.py
# Assumes each line of the file looks like:
#
#   n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
#
# where each line corresponds to one bounding box annotation associated
# with an image. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an image file.
arg_parser.add_argument('--bounding_box_file', dest = 'bounding_box_file', \
default = './imagenet_2012_bounding_boxes.csv', help = 'Bounding box file')

ARGS = arg_parser.parse_args()

def _int32_feature(value):
  """Wrapper for inserting int32 features into Example proto."""
  if not isinstance(value, list): 
    value = [value]
  return of_record.Feature(int32_list=of_record.Int32List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return of_record.Feature(float_list=of_record.FloatList(value=value))

def _double_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return of_record.Feature(double_list=of_record.DoubleList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  # if isinstance(value, six.string_types):
  #  value = six.binary_type(value, encoding='utf-8')
  return of_record.Feature(bytes_list=of_record.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, index, synset, human, bbox,
                        height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  for b in bbox:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    # pylint: enable=expression-not-assigned

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = of_record.OFRecord(feature={
      'data_id': _bytes_feature(str(index).encode('utf-8')),
      'height': _int32_feature(height),
      'width': _int32_feature(width),
      'colorspace': _bytes_feature(colorspace.encode('utf-8')),
      'channels': _int32_feature(channels),
      'class/label': _int32_feature(label),
      'class/synset': _bytes_feature(synset.encode('utf-8')),
      'class/text': _bytes_feature(human.encode('utf-8')),
      'object/bbox/xmin': _float_feature(xmin),
      'object/bbox/xmax': _float_feature(xmax),
      'object/bbox/ymin': _float_feature(ymin),
      'object/bbox/ymax': _float_feature(ymax),
      'object/bbox/label': _int32_feature([label] * len(xmin)),
      'format': _bytes_feature(image_format.encode('utf-8')),
      'filename': _bytes_feature(os.path.basename(filename).encode('utf-8')),
      'encoded': _bytes_feature(image_buffer)})
  return example


class ImageCoder(object):
  """Helper class that provides image coding utilities."""

  def __init__(self,size = None):
    self.size = size
    
  def _resize(self, image_data):
    if self.size != None and image_data.shape[:2] != self.size:
      return cv2.resize(image_data, self.size)
    return image_data

  def image_to_jpeg(self, image_data, resize=ARGS.resize):
    # image_data = cv2.imdecode(np.fromstring(image_data, np.uint8), 1) # deprecated,
    image_data = cv2.imdecode(np.frombuffer(image_data, np.uint8), 1)
    if resize:
      image_data = self._resize(image_data)
    return cv2.imencode(".jpg", image_data)[1].tobytes(), image_data.shape[0], image_data.shape[1]

def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with open(filename, 'rb') as f:
    image_data = f.read()
    image_data,height, width =  coder.image_to_jpeg(image_data)
    #print(height, width)
    return image_data,height, width
    # Decode the RGB JPEG.
    # image_data = coder._resize(image_data)



def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, indexs, humans, bboxes, num_shards):
  """Processes and saves list of images as OFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0

  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    # output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_filename = 'part-%.5d' % (shard)
    output_file = os.path.join(ARGS.output_directory, output_filename)
   
    f = open(output_file, 'wb')
    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      synset = synsets[i]
      human = humans[i]
      bbox = bboxes[i]
      index = indexs[i]
      try:
        image_buffer, height, width = _process_image(filename, coder)
      except:
        print(filename)
        continue

      # print('filename, label, index,synset, human, bbox,height, width >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n',
      #       filename , label, index,synset, human, bbox,height, width)
      example = _convert_to_example(filename, image_buffer, label, index,
                                    synset, human, bbox,
                                    height, width)
      l = example.ByteSize()
      f.write(struct.pack("q", l))
      f.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    f.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()



def _process_image_files(name, filenames, synsets, labels, indexs, humans,
                         bboxes, num_shards):
  """Process and save list of images as OFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(synsets)
  assert len(filenames) == len(labels)
  assert len(filenames) == len(humans)
  assert len(filenames) == len(bboxes)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), ARGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])
  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (ARGS.num_threads, ranges))
  sys.stdout.flush()


  # Create a generic utility for converting all image codings.
  if ARGS.width <= 0 or ARGS.height <= 0:
    coder = ImageCoder()
  else:
    coder = ImageCoder((ARGS.width, ARGS.height))

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            synsets, labels, indexs, humans, bboxes, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  for t in threads:
    t.join()
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.

        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

      where 'n01440764' is the unique synset label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.

      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in 
                       open(labels_file, 'r').readlines()]

  labels = []
  filenames = []
  synsets = []

  # Leave label index 0 empty as a background class.
  label_index = 0# use to be 1

  # Construct the list of JPEG files and labels.
  for synset in challenge_synsets:
    if not os.path.exists(os.path.join(data_dir, synset)):
      continue

    jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
    matching_files = glob.glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    synsets.extend([synset] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
    label_index += 1
  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved OFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets), data_dir))
  return filenames, synsets, labels, shuffled_index


def _find_human_readable_labels(synsets, synset_to_human):
  """Build a list of human-readable labels.

  Args:
    synsets: list of strings; each string is a unique WordNet ID.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'

  Returns:
    List of human-readable strings corresponding to each synset.
  """
  humans = []
  for s in synsets:
    assert s in synset_to_human, ('Failed to find: %s' % s)
    humans.append(synset_to_human[s])
  return humans


def _find_image_bounding_boxes(filenames, image_to_bboxes):
  """Find the bounding boxes for a given image file.

  Args:
    filenames: list of strings; each string is a path to an image file.
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  Returns:
    List of bounding boxes for each image. Note that each entry in this
    list might contain from 0+ entries corresponding to the number of bounding
    box annotations for the image.
  """
  num_image_bbox = 0
  bboxes = []
  for f in filenames:
    basename = os.path.basename(f)
    if basename in image_to_bboxes:
      bboxes.append(image_to_bboxes[basename])
      num_image_bbox += 1
    else:
      bboxes.append([])
  print('Found %d images with bboxes out of %d images' % (
      num_image_bbox, len(filenames)))
  return bboxes


def _process_dataset(name, directory, num_shards, synset_to_human,
                     image_to_bboxes):
  """Process a complete data set and save it as a OFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  """
  filenames, synsets, labels, indexs = _find_image_files(directory, ARGS.labels_file)
  # ./train/n03085013/n03085013_23287.JPEG n03085013 508 652481

  humans = _find_human_readable_labels(synsets, synset_to_human)
  bboxes = _find_image_bounding_boxes(filenames, image_to_bboxes)

  _process_image_files(name, filenames, synsets, labels, indexs,
                       humans, bboxes, num_shards)


def _build_synset_lookup(imagenet_metadata_file):
  """Build lookup for synset to human-readable label.

  Args:
    imagenet_metadata_file: string, path to file containing mapping from
      synset to human-readable label.

      Assumes each line of the file looks like:

        n02119247    black fox
        n02119359    silver fox
        n02119477    red fox, Vulpes fulva

      where each line corresponds to a unique mapping. Note that each line is
      formatted as <synset>\t<human readable label>.

  Returns:
    Dictionary of synset to human labels, such as:
      'n02119022' --> 'red fox, Vulpes vulpes'
  """
  lines = open(imagenet_metadata_file, 'r').readlines()
  synset_to_human = {}
  for l in lines:
    if l:
      parts = l.strip().split('\t')
      assert len(parts) == 2
      synset = parts[0]
      human = parts[1]
      synset_to_human[synset] = human
  return synset_to_human


def _build_bounding_box_lookup(bounding_box_file):
  """Build a lookup from image file to bounding boxes.

  Args:
    bounding_box_file: string, path to file with bounding boxes annotations.

      Assumes each line of the file looks like:

        n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

      where each line corresponds to one bounding box annotation associated
      with an image. Each line can be parsed as:

        <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

      Note that there might exist mulitple bounding box annotations associated
      with an image file. This file is the output of process_bounding_boxes.py.

  Returns:
    Dictionary mapping image file names to a list of bounding boxes. This list
    contains 0+ bounding boxes.
  """
  lines = open(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes


def main():
  assert not ARGS.shards % ARGS.num_threads, (
      'Please make the ARGS.num_threads commensurate with ARGS.shards')

  print('Saving results to %s' % ARGS.output_directory)
  if not os.path.exists(ARGS.output_directory):
    os.makedirs(ARGS.output_directory)

  # Build a map from synset to human-readable label.
  synset_to_human = _build_synset_lookup(ARGS.imagenet_metadata_file)
  image_to_bboxes = _build_bounding_box_lookup(ARGS.bounding_box_file)

  # Run it!
  if ARGS.name == 'validation':
    _process_dataset('validation', ARGS.validation_directory,
                   ARGS.shards, synset_to_human, image_to_bboxes)
  else:
    _process_dataset('train', ARGS.train_directory, ARGS.shards,
                   synset_to_human, image_to_bboxes)


if __name__ == '__main__':
  main()