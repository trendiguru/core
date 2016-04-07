#!/usr/bin/env python
import sys
#sys.path.insert(0, 'home/yonatan/core')
from background_removal import get_image

base_image = get_image()
print base_image
#face_image = core/background_removal.find_face_cascade(base_image)
#print face_image