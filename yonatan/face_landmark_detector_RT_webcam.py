# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/core/yonatan/shape_predictor_68_face_landmarks.dat")

eyes_landmarks = {38, 39, 41, 42, 44, 45, 47, 48}

eyes_dict = {}

# initialize the video stream and allow the cammera sensor to warmup
# print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=-1 > 0).start()
time.sleep(2.0)

refObj = None

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # show the frame
    if refObj is None:
        cv2.imshow("Frame", frame)
    else:
        cv2.imshow("Frame", frame_2)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("d"):
        for j, (x, y) in enumerate(shape):
            if j + 1 in eyes_landmarks:
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
                eyes_dict[j+1] = (x, y)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        left_eye_x, left_eye_y = int(np.mean([eyes_dict[38][0], eyes_dict[39][0], eyes_dict[41][0], eyes_dict[42][0]])), int(np.mean([eyes_dict[38][1], eyes_dict[39][1], eyes_dict[41][1], eyes_dict[42][1]]))
        right_eye_x, right_eye_y = int(np.mean([eyes_dict[44][0], eyes_dict[45][0], eyes_dict[47][0], eyes_dict[48][0]])), int(np.mean([eyes_dict[44][1], eyes_dict[45][1], eyes_dict[47][1], eyes_dict[48][1]]))

        cv2.circle(frame, (left_eye_x, left_eye_y), 1, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye_x, right_eye_y), 1, (0, 255, 0), -1)

        left_eye = np.array((left_eye_x, left_eye_y))
        right_eye = np.array((right_eye_x, right_eye_y))

        # print "distance between eyes: {}".format(np.linalg.norm(right_eye - left_eye))






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

        # sort the contours from left-to-right and, then initialize the
        # distance colors and reference object
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
                  (255, 0, 255))
        refObj = None

        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 200:
                continue

            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)

            # compute the center of the bounding box
            cX = np.average(box[:, 0])
            cY = np.average(box[:, 1])

            # if this is the first contour we are examining (i.e.,
            # the left-most contour), we presume this is the
            # reference object
            if refObj is None:
                # unpack the ordered bounding box, then compute the
                # midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-right and
                # bottom-right
                (tl, tr, br, bl) = box
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # compute the Euclidean distance between the midpoints,
                # then construct the reference object
                D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                refObj = (box, (cX, cY), D / 3.370)

                frame_2 = frame.copy()

                cv2.drawContours(frame_2, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

                break

        # # draw the contours on the image
        # orig = frame.copy()
        # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

        # # stack the reference coordinates and the object coordinates
        # # to include the object center
        # refCoords = np.vstack([refObj[0], refObj[1]])
        # objCoords = np.vstack([box, (cX, cY)])

        # # loop over the original points
        # for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
        #     # draw circles corresponding to the current points and
        #     # connect them with a line
        #     cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
        #     cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
        #     cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
        #              color, 2)

        # compute the Euclidean distance between the coordinates,
        # and then convert the distance in pixels to distance in
        # units
        D = dist.euclidean(left_eye, right_eye) / refObj[2]
        # (mX, mY) = midpoint((xA, yA), (xB, yB))
        # cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


        # # show the output image
        # cv2.imshow("Image", orig)
        # cv2.waitKey(0)

        print "distance between eyes: {}".format(D)

    if key == ord("r"):
        refObj = None

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
