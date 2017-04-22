# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours
from imutils import perspective
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

image = imutils.rotate_bound(image, 90)

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # detect circles in the image
# circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 6, 100, param1=200, param2=1, maxRadius=15)

gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


# not(cv2.isContourConvex(c)

contour_counter = 0
# all_cY = np.zeros(len(cnts))
all_cY_dict = {}
new_cnts = np.ones(len(cnts))
patches = np.zeros((len(cnts), 4))
patch_counter = 0

index = {}
images = {}


print "len(cnts): {0}".format(len(cnts))

# print "cnts: {0}".format(cnts)

# loop over the contours individually
for j, c in enumerate(cnts):
    # if the contour is not sufficiently large, ignore it
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    if cv2.contourArea(c) > 1000 or cv2.contourArea(c) < 200 or len(approx) < 11:
        new_cnts[j] = 0
        continue
    else:
        print "c: {0}".format(c)
        patch_counter += 1
        box = cv2.minAreaRect(c)
        # print "c: {0}, cv2.contourArea(c): {1}".format(c, cv2.contourArea(c))
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        print "box: {0}".format(box)

        x, y, w, h = cv2.boundingRect(c)
        print "x, y, w, h: {0} {1} {2} {3}".format(x, y, w, h)
        patches[patch_counter - 1] = x, y, w, h
        # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # all_cY[contour_counter - 1] = cY
        all_cY_dict[j] = cY

        print "cX: {0}, cY: {1}".format(cX, cY)

print "new_cnts: {0}".format(new_cnts)

print "len(patches): {0}".format(len(patches))

# patches = patches[np.nonzero(patches)]

patches = patches[~np.all(patches == 0, axis=1)]

new_cnts = np.transpose(np.nonzero(new_cnts))

print "new_cnts: {0}".format(new_cnts)
print "len(patches): {0}".format(len(patches))
print "patches: {0}".format(patches)

# for patches



# for key, value in all_cY_dict.iteritems():
#
#
#     if all_cY_dict.values()[]

# hist, bin_edges = np.histogram(all_cY_dict.values(), bins=3, density=False)
#
# print "hist: {0}".format(hist)
#
#
# for bin in hist:
#     if bin == 1:
#         print "HHHHHHHHH"
#         new_cnts

# new_cnts = np.trim_zeros(new_cnts)


# loop over the contours individually
for idx in new_cnts:
    c = cnts[idx]
    # if the contour is not sufficiently large, ignore it
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    if cv2.contourArea(c) > 1000 or cv2.contourArea(c) < 200 or len(approx) < 11:
        continue
    else:
        cv2.drawContours(output, c, -1, (0, 128, 255), 8)
        contour_counter += 1
        print contour_counter

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # # order the points in the contour such that they appear
        # # in top-left, top-right, bottom-right, and bottom-left
        # # order, then draw the outline of the rotated bounding
        # # box
        # box = perspective.order_points(box)

        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])

        # all_cY[contour_counter - 1] = cY

        # all_cY = np.trim_zeros(all_cY)

        # print "all_cY: {0}, type(all_cY): {1}".format(all_cY, type(all_cY))

        print "cX: {0}, cY: {1}".format(cX, cY)

        # hist = cv2.calcHist([image], [0], c, [256], [0, 256])
        # print "hist: {0}".format(hist)

# all_cY = np.trim_zeros(all_cY)

# hist, bin_edges = np.histogram(all_cY, bins=3, density=False)

# print "hist: {0}".format(hist)

# all_cY = np.sort(all_cY)

# for i, coorY in enumerate(all_cY):
#     if all_cY[i] > all_cY[i+1] + 50 or all_cY[i] < all_cY[i+1] + 50

# print "all_cY: {0}, type(all_cY): {1}".format(all_cY, type(all_cY))

# show the output image
cv2.imshow("output", np.hstack([image, output]))
cv2.waitKey(0)

