__author__ = 'Nadav Paz'
# Libraries import
# TODO - combine pose-estimation face detection as a backup to the cascades face detection

import string
from Tkinter import Tk
from tkFileDialog import askopenfilename
import collections
import os

import cv2
import numpy as np

from . import constants
from . import Utils
from . import ccv_facedetector as ccv


def image_is_relevant(image, use_caffe=False, image_url=None):
    """
    main engine function of 'doorman'
    :param image: nXmX3 dim ndarray representing the standard resized image in BGR colormap
    :return: namedtuple 'Relevance': has 2 fields:
                                                    1. isRelevant ('True'/'False')
                                                    2. faces list sorted by relevance (empty list if not relevant)
    Thus - the right use of this function is for example:
    - "if image_is_relevant(image).is_relevant:"
    - "for face in image_is_relevant(image).faces:"
    """
    Relevance = collections.namedtuple('relevance', 'is_relevant faces')
    faces_dict = find_face_cascade(image, 10)
    if len(faces_dict['faces']) == 0:
        faces_dict = find_face_ccv(image, 10)
    if not faces_dict['are_faces']:
        # if use_caffe:
        # return Relevance(caffeDocker_test.is_person_in_img('url', image_url).is_person, [])
        # else:
        return Relevance(False, [])
    else:
        if len(faces_dict['faces']) > 0:
            return Relevance(True, faces_dict['faces'])
        else:
            return Relevance(False, [])


def find_face_ccv(image_arr, max_num_of_faces=100):
    if not isinstance(image_arr, np.ndarray):
        raise IOError('find_face got a bad input: not np.ndarray')
    else:  # do ccv
        faces = ccv.ccv_facedetect(image_array=image_arr)
        if faces is None or len(faces) == 0:
            return {'are_faces': False, 'faces': []}
        else:
            return {'are_faces': True, 'faces': choose_faces(image_arr, faces, max_num_of_faces)}


def find_face_cascade(image, max_num_of_faces=10):
    gray = cv2.cvtColor(image, constants.BGR2GRAYCONST)
    face_cascades = [
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt2.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt_tree.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_default.xml'))]
    cascade_ok = False
    for cascade in face_cascades:
        if not cascade.empty():
            cascade_ok = True
            break
    if cascade_ok is False:
        raise IOError("no good cascade found!")
    faces = []
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(5, 5),
            flags=constants.scale_flag
        )
        if len(faces) > 0:
            break
    if len(faces) == 0:
        return {'are_faces': False, 'faces': []}
    return {'are_faces': True, 'faces': choose_faces(image, faces, max_num_of_faces)}


def choose_faces(image, faces_list, max_num_of_faces):
    h, w, d = image.shape
    x_origin = int(w / 2)
    y_origin = int(0.125 * h)
    if not isinstance(faces_list, list):
        faces_list = faces_list.tolist()
    relevant_faces = []
    for face in faces_list:
        if face_is_relevant(image, face):
            dx = abs(face[0] + (face[2] / 2) - x_origin)
            dy = abs(face[1] + (face[3] / 2) - y_origin)
            position = 0.6 * np.power(np.power(0.4 * dx, 2) + np.power(0.6 * dy, 2), 0.5)
            size = 0.4 * abs((float(face[2]) - 0.1 * np.amax((h, w))))
            face_relevance = position + size
            face.append(face_relevance)
            relevant_faces.append(face)
    if len(relevant_faces) > 0:
        sorted_list = np.array(sorted(relevant_faces, key=lambda face: face[4]), dtype=np.uint16)
        return sorted_list[0:min((max_num_of_faces, len(sorted_list))), 0:4]
    else:
        return relevant_faces


def face_is_relevant(image, face):
    x, y, w, h = face
    # threshold = face + 4 faces down = 5 faces
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    face_ycrcb = ycrcb[y:y + h, x:x + w, :]
    if 0.05 * image.shape[0] < h < 0.25 * image.shape[0] \
            and y < (image.shape[0] / 2) - h \
            and is_skin_color(face_ycrcb):
        return True
    else:
        return False


def is_skin_color(face_ycrcb):
    h, w, d = face_ycrcb.shape
    num_of_skin_pixels = 0
    for i in range(0, h):
        for j in range(0, w):
            cond = face_ycrcb[i][j][0] > 0 and 131 < face_ycrcb[i][j][1] < 180 and 80 < face_ycrcb[i][j][2] < 130
            if cond:
                num_of_skin_pixels += 1
    return num_of_skin_pixels / float(h * w) > 0.33


def average_bbs(bb1, bb2):
    bb_x = int((bb1[0] + bb2[0]) / 2)
    bb_y = int((bb1[1] + bb2[1]) / 2)
    bb_w = int((bb1[2] + bb2[2]) / 2)  # this isn't necessarily width, it could be x2 if rect is [x1,y1,x2,y2]
    bb_h = int((bb1[3] + bb2[3]) / 2)

    bb_out = [bb_x, bb_y, bb_w, bb_h]
    return bb_out
    # bb_out = int(np.divide(bb1[:]+bb2[:],2))


def combine_overlapping_rectangles(bb_list):
    if len(bb_list) < 2:
        return bb_list
    iou_threshold = 0.8  # TOTALLY ARBITRARY THRESHOLD
    for i in range(0, len(bb_list)):
        for j in range(i + 1, len(bb_list)):
            bb1 = bb_list[i]
            bb2 = bb_list[j]
            iou = Utils.intersectionOverUnion(bb1, bb2)
            if iou > iou_threshold:
                print('combining bbs')
                bb_new = average_bbs(bb1, bb2)
                # bb_list.remove(bb1)
                print('bblist before ' + str(bb_list))
                bb_list = np.delete(bb_list, j, axis=0)
                bb_list = np.delete(bb_list, i, axis=0)
                bb_list = np.append(bb_list, bb_new, axis=0)
                print('bblist after ' + str(bb_list))

                return (combine_overlapping_rectangles(bb_list))
            else:
                print('iou too small, taking first bb')
                print('bblist before ' + str(bb_list))
                bb_list = np.delete(bb_list, j, axis=0)
                print('bblist after ' + str(bb_list))
                return (combine_overlapping_rectangles(bb_list))

    return (bb_list)


def body_estimation(image, face):
    x, y, w, h = face
    y_down = image.shape[0] - 1
    x_back = np.max([x - 2 * w, 0])
    x_back_near = np.max([x - w, 0])
    x_ahead = np.min([x + 3 * w, image.shape[1] - 1])
    x_ahead_near = np.min([x + 2 * w, image.shape[1] - 1])
    rectangles = {"BG": [], "FG": [], "PFG": [], "PBG": []}
    rectangles["FG"].append([x, x + w, y, y + h])  # face
    rectangles["PFG"].append([x, x + w, y + h, y_down])  # body
    rectangles["BG"].append([x, x + w, 0, y])  # above face
    rectangles["BG"].append([x_back, x, 0, y + h])  # head left
    rectangles["BG"].append([x + w, x_ahead, 0, y + h])  # head right
    rectangles["PFG"].append([x_back_near, x, y + h, y_down])  # left near
    rectangles["PFG"].append([x + w, x_ahead_near, y + h, y_down])  # right near
    if x_back_near > 0:
        rectangles["PBG"].append([x_back, x_back_near, y + h, y_down])  # left far
    if x_ahead_near < image.shape[1] - 1:
        rectangles["PBG"].append([x_ahead_near, x_ahead, y + h, y_down])  # right far
    return rectangles


def bb_mask(image, bounding_box):
    if isinstance(bounding_box, basestring):
        bb_array = [int(bb) for bb in string.split(bounding_box)]
    else:
        bb_array = bounding_box
    image_w = image.shape[1]
    image_h = image.shape[0]
    x, y, w, h = bb_array
    y_down = np.min([image_h-1, y+1.2*h])
    x_back = np.max([x-0.2*w, 0])
    y_up = np.max([0, y-0.2*h])
    x_ahead = np.min([image_w-1, x+1.2*w])
    rectangles = {"BG": [], "FG": [], "PFG": [], "PBG": []}
    rectangles["PFG"].append([x, x+w, y, y+h])
    rectangles["PBG"].append([x_back, x_ahead, y_up, y_down])
    mask = create_mask_for_gc(rectangles, image)
    return mask


def paperdoll_item_mask(item_mask, bb):
    x, y, w, h = bb
    mask_h, mask_w = item_mask.shape
    mask = np.zeros(item_mask.shape, dtype=np.uint8)
    y_down = np.min([mask_h - 1, y + 1.1 * h])
    x_back = np.max([x - 0.1 * w, 0])
    y_up = np.max([0, y - 0.1 * h])
    x_ahead = np.min([mask_w - 1, x + 1.1 * w])
    mask[y_up:y_down, x_back:x_ahead] = 3
    mask = np.where(item_mask != 0, 1, mask)
    return mask


def create_mask_for_gc(rectangles, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for rectangle in rectangles["BG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 0
    for rectangle in rectangles["PBG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 2
    for rectangle in rectangles["PFG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 3
    for rectangle in rectangles["FG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 1
    return mask


def create_arbitrary(image):
    h, w = image.shape[:2]
    mask = np.zeros([h, w], dtype=np.uint8)
    sub_h = h / 20
    sub_w = w / 10
    mask[2 * sub_h:18 * sub_h, 2 * sub_w:8 * sub_w] = 2
    mask[4 * sub_h:16 * sub_h, 3 * sub_w:7 * sub_w] = 3
    mask[7 * sub_h:13 * sub_h, 4 * sub_w:6 * sub_w] = 1
    return mask


def standard_resize(image, max_side):
    original_w = image.shape[1]
    original_h = image.shape[0]
    if image.shape[0] < max_side and image.shape[1] < max_side:
        return image, 1
    aspect_ratio = float(np.amax((original_w, original_h))/float(np.amin((original_h, original_w))))
    resize_ratio = float(float(np.amax((original_w, original_h))) / max_side)
    if original_w >= original_h:
        new_w = max_side
        new_h = max_side/aspect_ratio
    else:
        new_h = max_side
        new_w = max_side/aspect_ratio
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
    rect = (0, 0, image.shape[1]-1, image.shape[0]-1)
    bgdmodel = np.zeros((1, 65), np.float64)  # what is this wierd size about? (jr)
    fgdmodel = np.zeros((1, 65), np.float64)

    # bounding box was sent from a human - grabcut with bounding box mask
    if Utils.legal_bounding_box(bounding_box):
        if Utils.all_inclusive_bounding_box(image, bounding_box):  # bb is nearly the whole image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
        else:
            mask = bb_mask(image, bounding_box)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    # grabcut on the whole image, with/without face
    else:
        faces_dict = find_face_cascade(image)
        # if len(faces_dict['faces']) > 0:  # grabcut with mask
        #     try:
        #         rectangles = body_estimation(image, faces_dict['faces'][0])
        #         mask = create_mask_for_gc(rectangles, image)
        #     except:
        #         mask = create_mask_for_gc(image)
        #
        # else:  # grabcut with arbitrary rect
        mask = create_arbitrary(image)
        cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    return mask2


def get_masked_image(image, mask):
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def image_white_bckgnd(image, mask):
    for i in range(0, np.shape(image)[0]):
        for j in range(0, np.shape(image)[1]):
            if mask[i][j] == 0:
                image[i][j][0] = 255
                image[i][j][1] = 255
                image[i][j][2] = 255
    return image


def get_binary_bb_mask(image, bb=None):
    """
    The function returns a ones mask within the bb regions, and an image-size ones matrix in case of None bb
    :param image:
    :param bb:
    :return:
    """
    if (bb is None) or (bb == np.array([0, 0, 0, 0])).all():
        return np.ones((image.shape[1], image.shape[0]))
    x, y, w, h = bb
    bb_masked = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    bb_masked[y:y+h, x:x+w] = 255
    return bb_masked


def get_image():
    Tk().withdraw()
    filename = askopenfilename()
    big_image = cv2.imread(filename)
    return big_image


def face_skin_color_estimation(image, face_rect):
    x, y, w, h = face_rect
    face_image = image[y:y + h, x:x + w, :]
    face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    bins = 180
    n_pixels = face_image.shape[0] * face_image.shape[1]
    hist_hue = cv2.calcHist([face_hsv], [0], None, [bins], [0, 180])
    hist_hue = np.divide(hist_hue, n_pixels)
    skin_hue_list = []
    for l in range(0, 180):
        if hist_hue[l] > 0.013:
            skin_hue_list.append(l)
    return skin_hue_list


def simple_mask_grabcut(image, mask):
    rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    return mask2


def person_isolation(image, face):
    x, y, w, h = face
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
    return image_copy
