#!/usr/bin/env python
import caffe
import numpy as np
from trendi import background_removal, Utils
from PIL import Image
import cv2

path = "/home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
#image = Utils.get_cv2_img_array(path)
#im = Image.open(path)
#print image.shape

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




#base_image = np.array([caffe.io.load_image(path)])
#print base_image.shape
face_image = background_removal.find_face_cascade(image)
print face_image
'''

def find_face(image):
    gray = cv2.cvtColor(image, constants.BGR2GRAYCONST)
    face_cascades = [
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt2.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt_tree.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_default.xml'))]
    cascade_ok = False
    for cascade in face_cascades:
        if not cascade.empty():
            cascade_ok = True
            break
    if cascade_ok is False:
        raise IOError("no good cascade found!")
    faces = []
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(5, 5),
            flags=constants.scale_flag
        )
        if len(faces) > 0:
            break
    return faces


face = find_face(path)
print face