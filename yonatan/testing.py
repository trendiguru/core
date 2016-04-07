#!/usr/bin/env python
import caffe
import numpy as np
from trendi import background_removal

path = "/home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
base_image = np.array([caffe.io.load_image(path)])
#print base_image.shape
face_image = background_removal.find_face_cascade(base_image)
print face_image