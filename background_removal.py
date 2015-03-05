__author__ = 'Nadav Paz'
# Libraries import


import cv2
import numpy as np
import string


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
    mask = face_mask(rectangles, image)
    return mask


def face_mask(rectangles, image):
    GC_MASK_VALUES = {"BG": 0, "FG": 1, "PBG": 2, "PFG": 3}
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for key, rectangle in rectangles.iteritems():
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = GC_MASK_VALUES[key]
    return mask


def standard_resize(image, max_side):
    original_w = image.shape[1]
    original_h = image.shape[0]
    if image.shape[0] < 400 and image.shape[1] < 400:
        return image, 1
    resize_ratio = float(np.amax((original_w, original_h))) / np.min((original_h, original_w))
    if original_w >= original_h:
        new_w = max_side
        new_h = max_side/resize_ratio
    else:
        new_h = max_side
        new_w = max_side/resize_ratio
    resized_image = cv2.resize(image, (int(new_w), int(new_h)))
    return resized_image, resize_ratio


def resize_back(image, resize_ratio):
    w = image.shape[1]
    h = image.shape[0]
    new_w = w*resize_ratio
    new_h = h*resize_ratio
    resized_image = cv2.resize(image, (int(new_w), int(new_h)))
    return resized_image


def get_fg_mask(image, bounding_box=None):
    # image_counter = 0
    rect = (0, 0, image.shape[1]-1, image.shape[0]-1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    # bounding box was sent from a human - grabcut with bounding box mask
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        mask = bb_mask(image, bounding_box)
        cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    # grabcut on the whole image, with/without face
    else:
        face = find_face(image)
        if len(face) > 0:                                # grabcut with mask
            rectangles = body_estimation(image, face)
            mask = face_mask(rectangles, image)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
            # album.append(cv2.bitwise_and(image, image, mask=mask2))
            # image_counter += 1
        else:                                             # grabcut with rect
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    return mask2


def get_masked_image(image, mask):
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def combine_mask_and_bb(masked_image, bb):
    x = bb[0]
    y = bb[1]
    w = bb[2]
    h = bb[3]
    combined_image = masked_image[y:y+h, x:x+w]
    return combined_image


def get_bb_mask(image, bb):
    x, y, w, h = bb
    bb_masked = np.zeros(image.shape[0], image.shape[1])
    bb_masked[y:y+h, x:x+w] = 255
    return bb_masked