#!/usr/bin/env python
import caffe
import numpy as np
from trendi import background_removal, Utils
from PIL import Image

path = "/home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
image = Utils.get_cv2_img_array(path)
#im = Image.open(path)
print image.size

'''
im_rgb = Image.open(path).convert('RGB')
im_arr_rgb = np.array(im_rgb)
print im_arr_rgb.size


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True

print is_grey_scale(path)
'''



#base_image = np.array([caffe.io.load_image(path)])
#print base_image.shape
face_image = background_removal.find_face_cascade(image)
print face_image