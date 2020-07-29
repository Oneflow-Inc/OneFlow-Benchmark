import oneflow as flow
import numpy as np
import oneflow.core.record.record_pb2 as ofrecord
from PIL import Image
import struct
import cv2
import os

DIR = "data/facades/"


def _process_image_file(filename):
    img = cv2.imread(filename)
    real_img = img[:, :256, :]
    inp_img = img[:, 256:, :]
    ok, real_img = cv2.imencode(".jpg", real_img)
    ok, inp_img = cv2.imencode(".jpg", inp_img)
    return real_img.tobytes(), inp_img.tobytes()


def _bytes_feature(value):
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=[value]))


def _gen_example(real_img, input_img):
    example = ofrecord.OFRecord(
        feature={
            "input_img": _bytes_feature(input_img),
            "real_img": _bytes_feature(real_img),
        }
    )
    return example


def gen_ofrecord(mode):
    with open(os.path.join(DIR, "part-0"), "wb") as f:
        for d in os.listdir(os.path.join(DIR, mode)):
            filename = os.path.join(DIR, mode, d)
            real_img, input_img = _process_image_file(filename)
            example = _gen_example(real_img, input_img)
            l = example.ByteSize()
            f.write(struct.pack("q", l))
            f.write(example.SerializeToString())


def decoder(data_dir):
    ofrecord = flow.data.ofrecord_reader(data_dir,
                                         batch_size=1,
                                         data_part_num=1,
                                         part_name_suffix_length=1,
                                         random_shuffle=True,
                                         shuffle_after_epoch=True)
    inp_image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, 
                                                         "input_img", 
                                                         color_space='RGB',
                                                         random_area=[0.9, 1.0],
                                                         random_aspect_ratio=[1.0, 1.0])
    real_image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, 
                                                         "real_img", 
                                                         color_space='RGB',
                                                         random_area=[0.9, 1.0],
                                                         random_aspect_ratio=[1.0, 1.0])
    inp = flow.image.Resize(inp_image, resize_x=256,
                            resize_y=256, color_space='RGB')
    tar = flow.image.Resize(real_image, resize_x=256,
                            resize_y=256, color_space='RGB')
    return inp, tar


def test_ofrecord():
    gen_ofrecord('train')

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def OfrecordDecoderJob():
        inp, tar = decoder(DIR)
        return inp, tar

    inp, tar = OfrecordDecoderJob().get()
    cv2.imwrite("test.jpg", inp[0])
    cv2.imwrite("test1.jpg", tar[0])
    print(inp.numpy().shape)
    print(tar.numpy().shape)


if __name__ == "__main__":
    test_ofrecord()
    # gen_ofrecord('train')
