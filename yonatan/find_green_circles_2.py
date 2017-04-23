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

    cv2.drawContours(image_rotate, [c], -1, (0, 128, 255), 2)
    text_center = "(cX:{0}, cY:{1})".format(cX, cY)
    cv2.putText(image_rotate, text_center, (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # the radius of the contour is approximately half green and half blue,
    # and i want to find the area of the green circle, so i divide the radius by 2
    contour_area = np.pi * (0.5 * 0.5 * (box[3][0] - box[0][0]))**2  # px
    text_area = "area: {0}px".format(contour_area)
    cv2.putText(image_rotate, text_area, (cX, cY + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    ratio = 1
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)


    # # Create a mask holder
    # patch = image_rotate[box[0][0]:box[3][0], box[1][1]:box[0][1]]
    #
    # mask = np.zeros(patch.shape[:2], np.uint8)
    #
    # # Grab Cut the object
    # bgdModel = np.zeros((1, 65), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    #
    # try:
    #     # Hard Coding the Rect The object must lie within this rect.
    #     rect = (cX - 3, cY - 3, 6, 6)
    #     cv2.grabCut(patch, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    #     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #     # img1 = image_rotate[box[0][0]:box[3][0], box[1][1]:box[0][1]] * mask[:, :, np.newaxis]
    #
    #     cv2.drawContours(output, mask, -1, (0, 128, 255), 8)
    #
    #     cv2.imshow("output", np.hstack([image_rotate, output]))
    #     cv2.waitKey(0)
    #
    # except:
    #     continue

    # # define BGR boundaries
    # lower_green = np.array([0, 107, 0], dtype="uint8")
    # upper_green = np.array([50, 255, 154], dtype="uint8")
    #
    # mask_green = cv2.inRange(image_rotate[box[0][0]:box[3][0], box[1][1]:box[0][1]], lower_green, upper_green)
    #
    # if cv2.countNonZero(mask_green) / float((box[0][1] - box[1][1]) * (box[3][0] - box[0][0])) > 0.01:
    #     print "GREEN"
    #     cv2.drawContours(output, c, -1, (0, 128, 255), 8)
    #
    # cv2.imshow("output", np.hstack([image_rotate, output]))
    # cv2.waitKey(0)





    hist = cv2.calcHist([image_rotate[box[0][0]:box[3][0], box[1][1]:box[0][1]]], [1], None, [256],
                        [0, 256])  # the histogram is on the green channel ([1])

    if sum(hist) > 600:

        gray_patch = gray[box[0][0]:box[3][0], box[1][1]:box[0][1]]
        edged_patch = cv2.Canny(gray_patch, 50, 100)
        edged_patch = cv2.dilate(edged_patch, None, iterations=1)
        edged_patch = cv2.erode(edged_patch, None, iterations=1)

        # find contours in the edge map
        cnts_patch = cv2.findContours(edged_patch.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_patch = cnts_patch[0] if imutils.is_cv2() else cnts_patch[1]

        for c_p in cnts_patch:
            print "i'm in!"
            cv2.drawContours(image_rotate, c_p, 0, (0, 0, 0), -1)

            M_patch = cv2.moments(c_p)
            cX_p = int((M_patch["m10"] / M_patch["m00"]) * ratio)
            cY_p = int((M_patch["m01"] / M_patch["m00"]) * ratio)

            box_p = cv2.minAreaRect(c_p)
            box_p = cv2.cv.BoxPoints(box_p) if imutils.is_cv2() else cv2.boxPoints(box_p)
            box_p = np.array(box_p, dtype="int")

            contour_area_patch = np.pi * (0.5 * (box_p[3][0] - box_p[0][0])) ** 2  # px
            text_area = "area: {0}px".format(contour_area_patch)
            cv2.putText(image_rotate, text_area, (cX, cY - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        cv2.drawContours(output, c, -1, (0, 128, 255), 8)
        # print "cX: {0}, cY: {1}".format(cX, cY)

    # print "hist: {0}".format(hist)

# show the original image next to the output image
cv2.imshow("output", np.hstack([image_rotate, output]))
cv2.waitKey(0)

