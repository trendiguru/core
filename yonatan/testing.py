#!/usr/bin/env python

from trendi import background_removal

base_image = background_removal.get_image()
print base_image
face_image = background_removal.find_face_cascade(base_image)
print face_image