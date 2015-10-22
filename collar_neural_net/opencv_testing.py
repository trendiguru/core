
import numpy as np
import cv2

file = '13.jpg'

image = cv2.imread(file,1)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.imwrite('new_image.jpg',image)
