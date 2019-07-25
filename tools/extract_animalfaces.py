"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('imagenet_folder', type=str)
parser.add_argument('--output_folder', type=str)
parser.add_argument('--coor_file', type=str)
opts = parser.parse_args()
IMAGENET_TRAIN = opts.imagenet_folder
OUT_PUT_FOLDER = opts.output_folder
COOR_FILE = opts.coor_File

with open(COOR_FILE, 'rt') as f:
    lines = f.readlines()

for l in lines:
    ls = l.strip().split(' ')
    img_name = os.path.join(IMAGENET_TRAIN, ls[0])
    img = Image.open(img_name)
    img = img.convert('RGB')
    x = int(ls[1])
    y = int(ls[2])
    w = int(ls[3])
    h = int(ls[4])
    out_name = os.path.join(OUT_PUT_FOLDER,
                            '%s_%d_%d_%d_%d.jpg' % (ls[0], x, y, w, h))
    crop = img.crop((x, y, w, h))
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    print(out_name)
    crop.save(out_name)
