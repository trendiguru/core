import cv2
import numpy as np

height = 100
width = 500
blank_image = np.zeros((height, width, 3), np.uint8)
maxrange = 180
prev_x = 0
for i in range(0, maxrange + 1):
    x = int(i * width / maxrange)
    color = int(i * 180 / maxrange)
    blank_image[:, prev_x:x] = (color, 200, 200)  # (B, G, R)
    rgb = cv2.cvtColor(blank_image, cv2.COLOR_HSV2BGR)
    print('x {0} prev x {1}'.format(x, prev_x))
    prev_x = x

cv2.imshow('img', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)None