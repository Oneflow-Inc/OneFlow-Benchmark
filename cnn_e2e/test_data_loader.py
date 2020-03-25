import config as configs
parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

from dali_util import get_rec_iter
import numpy as np

import ofrecord_util
from job_function_util import get_val_config
import oneflow as flow

from PIL import Image

flow.config.gpu_device_num(args.gpu_num_per_node)
flow.config.enable_debug_mode(True)
@flow.function(get_val_config(args))
def InferenceNet():
    (labels, images) = ofrecord_util.load_imagenet_for_validation(args)
    return images, labels

def save_bmp(array, filepath):
    print(array.dtype)
    im = Image.fromarray(array)
    im.save(filepath)

if __name__ == '__main__':
    train_data_iter, val_data_iter = get_rec_iter(args, True)
    #train_data_iter.reset()
    for i, batches in enumerate(val_data_iter):
        images, labels = batches
        print(labels)
        np.save('output/dali_val_data.npy', images)
        save_bmp(images[0], 'output/dali_val.bmp')


    for i, batches in enumerate(train_data_iter):
        images, labels = batches
        print(labels)
        np.save('output/dali_train_data.npy', images)

    images, labels = InferenceNet().get()
    images = images.ndarray().astype(np.uint8)
    np.save('output/of_val_data.npy', images)
    save_bmp(images[0], 'output/of_val.bmp')
    