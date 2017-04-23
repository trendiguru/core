import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, resize, rotate, clone it for output, and then convert it to grayscale
image_orig = cv2.imread(args["image"])
image_resize = imutils.resize(image_orig, width=900)
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

# the indexes of the contours that qualify the condition will be in new_cnts
new_cnts = np.ones(len(cnts))

# loop over the contours
for j, c in enumerate(cnts):
    # if the contour don't stand in the conditions, ignore it
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    center, radius = cv2.minEnclosingCircle(c)
    if cv2.contourArea(c) > 1000 or cv2.contourArea(c) < 440 or len(approx) < 11 or radius > 20:
        new_cnts[j] = 0
        continue

# drop the zeros
new_cnts = np.transpose(np.nonzero(new_cnts))

# loop over the contours that made the cut
for idx in new_cnts:
    c = cnts[idx]

    # find the corners of the minimum box that can include the contour
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # find center coordinates of the contour
    center, radius = cv2.minEnclosingCircle(c)
    cX_float, cY_float = center
    cX, cY = int(cX_float), int(cY_float)

    # the patch in the original image of the box (of the contour)
    patch = image_rotate[box[1][1]:box[0][1], box[0][0]:box[3][0]]
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    # just a reminder: colors = ("b", "g", "r")
    chans = cv2.split(patch)

    # histogram of the green channel
    hist_green = cv2.calcHist([chans[1]], [0], None, [256], [0, 256])
    # histogram of the blue channel
    hist_blue = cv2.calcHist([chans[0]], [0], None, [256], [0, 256])

    compare_hist = cv2.compareHist(hist_green, hist_blue, cv2.cv.CV_COMP_CORREL)

    # empiric value
    if compare_hist > 0.20:
        cv2.drawContours(output, [c], -1, (0, 128, 255), 2)

        text_center = "(cX:{0}, cY:{1})".format(cX, cY)
        cv2.putText(output, text_center, (cX - 45, cY + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        contour_area = cv2.contourArea(c) * 0.5  # px
        text_area = "green_area: {0}px".format(contour_area)
        cv2.putText(output, text_area, (cX - 45, cY + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# show the original image next to the output image
cv2.imshow("output", np.hstack([image_rotate, output]))
cv2.waitKey(0)

