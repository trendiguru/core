import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image_orig = cv2.imread(args["image"])
width = 900
image_resize = imutils.resize(image_orig, width=width)
image_rotate = imutils.rotate_bound(image_resize, 90)

output = image_rotate.copy()
gray = cv2.cvtColor(image_rotate, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

new_cnts = np.ones(len(cnts))

# loop over the contours
for j, c in enumerate(cnts):
    # if the contour don't stand in the conditions, ignore it
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    center, radius = cv2.minEnclosingCircle(c)
    if cv2.contourArea(c) > 1000 or cv2.contourArea(c) < 100 or len(approx) < 11 or radius > 20:
        new_cnts[j] = 0
        continue
    # else:
    #     ratio = 1
    #     # compute the center of the contour
    #     M = cv2.moments(c)
    #     cX = int((M["m10"] / M["m00"]) * ratio)
    #     cY = int((M["m01"] / M["m00"]) * ratio)
    #
    #     print "cX: {0}, cY: {1}".format(cX, cY)

new_cnts = np.transpose(np.nonzero(new_cnts))


# loop over the contours that made the cut
for idx in new_cnts:
    c = cnts[idx]

    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # compute the center of the bounding box
    cX = int(np.average(box[:, 0]))
    cY = int(np.average(box[:, 1]))

    print "cX: {0}, cY: {1}".format(cX, cY)

    cv2.drawContours(image_rotate, [c], -1, (0, 128, 255), 2)
    text_center = "(cX:{0}, cY:{1})".format(cX, cY)
    cv2.putText(image_rotate, text_center, (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    contour_area = cv2.contourArea(c)
    contour_area_2 = np.pi * (0.5 * (box[3][0] - box[0][0]))**2  # px

    print "koter: {}".format(box[3][0] - box[0][0])
    print "contour_area: {0}, contour_area_2: {1}".format(contour_area, contour_area_2)

    text_area = "area: {0}".format(contour_area)
    cv2.putText(image_rotate, text_area, (cX, cY + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    hist = cv2.calcHist([image_rotate[box[0][0]:box[3][0], box[1][1]:box[0][1]]], [1], None, [256],
                        [0, 256])

    if sum(hist) > 600:
        cv2.drawContours(output, c, -1, (0, 128, 255), 8)
        # print "cX: {0}, cY: {1}".format(cX, cY)

    # print "hist: {0}".format(hist)

# show the original image next to the output image
cv2.imshow("output", np.hstack([image_rotate, output]))
cv2.waitKey(0)

