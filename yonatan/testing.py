#!/usr/bin/env python
import caffe
import numpy as np
from trendi import background_removal
from PIL import Image

path = "/home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
im = Image.open(path)
print im.size

im_rgb = Image.open(path).convert('RGB')
print im_rgb.size

def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True

print is_grey_scale(path)




base_image = np.array([caffe.io.load_image(path)])
#print base_image.shape
face_image = background_removal.find_face_cascade(im)
print face_image