__author__ = 'jeremy'

import cv2

print('cv2 version is ' + str(cv2.__version__))
if cv2.__version__ == '3.0.0':
    print('cv2 version is 3')
    const = cv2.CASCADE_SCALE_IMAGE
else:
    const = cv2.cv.CV_HAAR_SCALE_IMAGE

