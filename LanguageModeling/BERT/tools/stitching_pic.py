import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image

def stiiching_pic(dir, out_file):

    hight = 1200 
    width = 3600
    file_lsit = os.listdir(dir)
    target = Image.new('RGBA', (width, hight*len(file_lsit)))
    left = 0
    right = hight
    for file in file_lsit:

        tmp = dir+'/'+file
        print(tmp)
        image = Image.open(tmp)
        # print(image)
        # print(target)
        target.paste(image, (0, left, width, right))
        left += hight 
        right += hight
    target.save(out_file)


def main():
    print('test')
    parser = argparse.ArgumentParser(description="Stitching pictures")
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()

    stiiching_pic(args.dir, args.out_file)


if __name__ == "__main__":
    main()
