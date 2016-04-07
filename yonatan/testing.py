#!/usr/bin/env python

from trendi import background_removal

base_image = "home/yonatan/test_set/female/Juljia_Vysotskij_0001.jpg"
print base_image.shape
face_image = background_removal.find_face_cascade(base_image)