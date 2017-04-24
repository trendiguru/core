''' object tracking, opencv
https://github.com/opencv/opencv_contrib/issues/640

'''
__author__ = 'jeremy'

import cv2
firstFrame = True
pathVid =  '/home/jeremy/Downloads/NWAC Soccer Championships- Mens Championship Game - Tacoma vs. Spokane.mp4'
# e.g. vidName.mp4 or images/%04.jpg (if image name is 0001.jpg...)
vidReader = cv2.VideoCapture(pathVid)
initBbox = (30, 60, 60, 70 )  # (x_tl, y_tl, w, h)
#tracker = cv2.Tracker_create('KCF')
tracker = cv2.Tracker_create('KCF')
while vidReader.isOpened():
    ok, image=vidReader.read()
    if firstFrame:
        ok = tracker.init(image, initBbox)
        firstFrame = False
    ok, bbox = tracker.update(image)
    print ok, bbox
    