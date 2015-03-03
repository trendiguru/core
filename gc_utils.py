__author__ = 'Nadav Paz'

import numpy as np
import string
import cv2


def find_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascades = [cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml'),
                     cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'),
                     cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml'),
                     cv2.CascadeClassifier('haarcascade_frontalface_default.xml')]
    for i in range(0, 3, 1):
        faces = face_cascades[i].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(1, 1),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if len(faces) > 0:
            break
    return faces


def body_estimation(image, face):
            x = face[0][0]
            y = face[0][1]
            w = face[0][2]
            h = face[0][3]
            y_down = image.shape[0]-1
            x_back = np.max([x-2*w, 0])
            x_ahead = np.min([x+3*w, image.shape[1]-1])
            rectangles = {"BG": [], "FG": [], "PFG": [], "PBG": []}
            rectangles["FG"].append([x, x+w, y, y+h])                   # face
            rectangles["PFG"].append([x, x+w, y+h, y_down])             # body
            rectangles["BG"].append([x, x+w, 0, y])                     # above face
            rectangles["BG"].append([x_back, x, 0, y+h])                # head left
            rectangles["BG"].append([x+w, x_ahead, 0, y+h])             # head right
            rectangles["PFG"].append([x-w, x, y+h, y_down])             # left near
            rectangles["PFG"].append([x+w, x+2*w, y+h, y_down])         # right near
            rectangles["PBG"].append([x_back, x-w, y+h, y_down])        # left far
            rectangles["PBG"].append([x+2*w, x_ahead, y+h, y_down])     # right far
            return rectangles


def face_mask(rectangles, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for rectangle in rectangles["BG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 0
    for rectangle in rectangles["FG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 1
    for rectangle in rectangles["PBG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 2
    for rectangle in rectangles["PFG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 3
    return mask


def bb_mask(image, bounding_box):
    if isinstance(bounding_box, basestring):
        bb_array = [int(bb) for bb in string.split(bounding_box)]
    else:
        bb_array = bounding_box
    image_w = image.shape[1]
    image_h = image.shape[0]
    x = bb_array[0]
    y = bb_array[1]
    w = bb_array[2]
    h = bb_array[3]
    rectangles = {"PFG": [], "PBG": []}
    rectangles["PFG"].append([x, x+w, y, y+h])
    rectangles["PBG"].append([0, image_w-1, 0, y])
    rectangles["PBG"].append([0, x, y, y+h])
    rectangles["PBG"].append([x+h, image_w-1, y, y+h])
    rectangles["PBG"].append([0, image_w-1, y+h, image_h-1])
    mask = build_mask(rectangles, image)
    return mask


def build_mask(rectangles, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for rectangle in rectangles["BG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 0
    for rectangle in rectangles["FG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 1
    for rectangle in rectangles["PBG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 2
    for rectangle in rectangles["PFG"]:
        x0 = rectangle[0]
        x1 = rectangle[1]
        y0 = rectangle[2]
        y1 = rectangle[3]
        mask[y0:y1, x0:x1] = 3
    return mask

