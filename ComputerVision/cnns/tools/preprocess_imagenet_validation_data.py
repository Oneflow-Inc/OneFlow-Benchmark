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
#!/usr/bin/python
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
"""Process the ImageNet Challenge bounding boxes for OneFlow model training.

Associate the ImageNet 2012 Challenge validation data set with labels.

The raw ImageNet validation data set is expected to reside in JPEG files
located in the following directory structure.

 data_dir/ILSVRC2012_val_00000001.JPEG
 data_dir/ILSVRC2012_val_00000002.JPEG
 ...
 data_dir/ILSVRC2012_val_00050000.JPEG

This script moves the files into a directory structure like such:
 data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
 data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
 ...
where 'n01440764' is the unique synset label associated with
these images.

Sample usage:
  python3 preprocess_imagenet_validation_data.py  ../data/imagenet/validation
"""


import os.path
import sys

from six.moves import xrange


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Invalid usage\n'
          'usage: preprocess_imagenet_validation_data.py '
          '<validation data dir>')
    sys.exit(-1)
  data_dir = sys.argv[1]
  validation_labels_file = "imagenet_2012_validation_synset_labels.txt"

# Read in the 50000 synsets associated with the validation data set.
labels = [l.strip() for l in open(validation_labels_file).readlines()]
unique_labels = set(labels)

# Make all sub-directories in the validation data dir.
for label in unique_labels:
  labeled_data_dir = os.path.join(data_dir, label)
  if not os.path.exists(labeled_data_dir):
    os.makedirs(labeled_data_dir)

# Move all of the image to the appropriate sub-directory.
for i in xrange(len(labels)):
  basename = 'ILSVRC2012_val_000%.5d.JPEG' % (i + 1)
  original_filename = os.path.join(data_dir, basename)
  if not os.path.exists(original_filename):
    continue
  print('Get image: ', original_filename)
  new_filename = os.path.join(data_dir, labels[i], basename)
  os.rename(original_filename, new_filename)


# Delete all empty dir
for label in unique_labels:
  labeled_data_dir = os.path.join(data_dir, label)
  if not os.path.exists(labeled_data_dir):
    continue
  if not os.listdir(labeled_data_dir):
    os.rmdir(labeled_data_dir)

