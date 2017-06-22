__author__ = 'jeremy'
# __author__ = 'jeremy'
#
# # import the necessary packages
# import numpy as np
# import argparse
# import cv2
#
# from trendi.defense import defense_client
#
# # initialize the current frame of the video, along with the list of
# # ROI points along with whether or not this is input mode
# frame = None
# roiPts = []
# inputMode = False
#
#
# def selectROI(event, x, y, flags, param):
#     # grab the reference to the current frame, list of ROI
#     # points and whether or not it is ROI selection mode
#     global frame, roiPts, inputMode
#
#     # if we are in ROI selection mode, the mouse was clicked,
#     # and we do not already have four points, then update the
#     # list of ROI points with the (x, y) location of the click
#     # and draw the circle
#     if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
#         roiPts.append((x, y))
#         cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
#         cv2.imshow("frame", frame)
#
#
# def main(vid='/home/jeremy/projects/soccer/UNC_Jordan_McCrary_Highlights.mp4'):
#     global frame, roiPts, inputMode
#
#     # construct the argument parse and parse the arguments
#     # ap = argparse.ArgumentParser()
#     # ap.add_argument("-v", "--video",
#     #     help = "path to the (optional) video file")
#     # args = vars(ap.parse_args())
#     #
#     # # grab the reference to the current frame, list of ROI
#     # # points and whether or not it is ROI selection mode
#     #
#     # # if the video path was not supplied, grab the reference to the
#     # # camera
#     # if not args.get("video", False):
#     #     camera = cv2.VideoCapture(0)
#     #
#     # # otherwise, load the video
#     # else:
#     #     camera = cv2.VideoCapture(args["video"])
#
#     camera = cv2.VideoCapture(vid)
#
#     # setup the mouse callback
#     cv2.namedWindow("frame")
#     cv2.setMouseCallback("frame", selectROI)
#
#     # initialize the termination criteria for cam shift, indicating
#     # a maximum of ten iterations or movement by a least one pixel
#     # along with the bounding box of the ROI
#     termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#     roiBox = None
# # keep looping over the frames
#     n_frame=0
#     while True:
#         n_frame=n_frame+1
#         # grab the current frame
#         print('frame {} '.format(n_frame))
#         (grabbed, frame) = camera.read()
#         # check to see if we have reached the end of the
#         # video
#         if not grabbed:
#             break
#
#         # if the see if the ROI has been computed
#         if roiBox is not None:
#             # convert the current frame to the HSV color space
#             # and perform mean shift
#             print('detecting ROI')
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
#
#             # apply cam shift to the back projection, convert the
#             # points to a bounding box, and then draw them
#             (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
#
#             pts = np.int0(cv2.boxPoints(r))
#             cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
#
#                 # show the frame and record if the user presses a key
#             print('roiBox {} roipts {} pts {}'.format(roiBox,roiPts,pts))
#
#         cv2.imshow("frame", frame)
#         key = cv2.waitKey(0) & 0xFF
#
#         # handle if the 'i' key is pressed, then go into ROI
#         # selection mode
#         if key == ord("i") :# and len(roiPts) < 4:
#             # indicate that we are in input mode and clone the
#             # frame
#             inputMode = True
#             orig = frame.copy()
#
#             # keep looping until 4 reference ROI points have
#             # been selected; press any key to exit ROI selction
#             # mode once 4 points have been selected
#             while len(roiPts) < 4:
#                 cv2.imshow("frame", frame)
#                 cv2.waitKey(0)
#
#             # determine the top-left and bottom-right points
#             roiPts = np.array(roiPts)
#             s = roiPts.sum(axis = 1)
#             tl = roiPts[np.argmin(s)]
#             br = roiPts[np.argmax(s)]
#
#             # grab the ROI for the bounding box and convert it
#             # to the HSV color space
#             roi = orig[tl[1]:br[1], tl[0]:br[0]]
#             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
#
#             # compute a HSV histogram for the ROI and store the
#             # bounding box
#             roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
#             roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
#             roiBox = (tl[0], tl[1], br[0], br[1])
#
#         # if the 'q' key is pressed, stop the loop
#         elif key == ord("q"):
#             break
#
#         elif key == ord("g"):
#             print('attempting to get answer from yolo')
#             res = defense_client.detect_hls(frame)
#             data = res['data']
#             print('from yolo got {}'.format(data))
#             if len(data)==0:
#                 print('no data in frame ')
#                 continue
#             data = sorted(data,key=lambda object:(-object['confidence']))
#             print('sorted data:'+str(data))
#
#             firstbox = data[0]['bbox']
#             tl = (firstbox[0],firstbox[1])
#             br = (firstbox[0]+firstbox[2],firstbox[1]+firstbox[3])
#
#
#             # grab the ROI for the bounding box and convert it
#             # to the HSV color space
#             orig = frame.copy()
#             roi = orig[tl[1]:br[1], tl[0]:br[0]]
#             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
#
#             # compute a HSV histogram for the ROI and store the
#             # bounding box
#             roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
#             roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
#             roiBox = (firstbox[0],firstbox[1],firstbox[0]+firstbox[2],firstbox[1]+firstbox[3])
#
#
#
#         # cleanup the camera and close any open windows
#     camera.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()




#!/usr/bin/env python

'''
Camshift tracker
================
This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)
http://www.robinhewitt.com/research/track/camshift.html
Usage:
------
    camshift.py [<video source>]
    To initialize tracking, select the object with mouse
Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2

# local module
# import video
# from video import presets

vid='/home/jeremy/projects/soccer/UNC_Jordan_McCrary_Highlights.mp4'
camera = cv2.VideoCapture(vid)


class App(object):
    def __init__(self, video_src):
        # self.cam = video.create_capture(video_src, presets['cube'])
        # ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

    def onmouse(self, event, x, y, flags, param):
        print('onmouse')
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        print('showhist')
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def run(self):
        n_frame=0
        while True:
            (grabbed, frame) = camera.read()
            n_frame+=1
            print('frame {} selection {}'.format(n_frame,self.selection))
        # check to see if we have reached the end of the

        # video
            if not grabbed:
                break
            self.frame  = frame
#            ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                print('selection')
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                print('track window')
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    pass
                    #vis[:] = prob[...,np.newaxis]
                try:
                    cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                except:
                    print(track_box)

            cv2.imshow('camshift', vis)

            ch = cv2.waitKey(0)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()



import cv2
import sys

def try_opencv_trackers():

    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    tracker = cv2.Tracker_create("MIL")

    # Read video
    video = cv2.VideoCapture("videos/chaplin.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break


if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    print(__doc__)
    a=App(video_src)
    a.run()
