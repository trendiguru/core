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
    cv2.imshow("Frame", frame)
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

        print "distance between eyes: {}".format(np.linalg.norm(right_eye - left_eye))

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
