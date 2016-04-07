#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/yonatan/core')
import background_removal.py

base_image = background_removal.get_image()
print base_image
#face_image = core/background_removal.find_face_cascade(base_image)
#print face_image