__author__ = 'Nadav Paz'
# Libraries import
# TODO throw error in find_face if xml's aren't found - currently i think this happens silently
# TODO - combine pose-estimation face detection as a backup to the cascades face detection

import string
from Tkinter import Tk
from tkFileDialog import askopenfilename
import collections
import os
import logging

import cv2
import numpy as np

import constants
import Utils


def image_is_relevant(image):
    """
    main engine function of 'doorman'
    :param image: nXmX3 dim ndarray representing the standard resized image in BGR colormap
    :return: namedtuple 'Relevance': has 2 fields:
                                                    1. isRelevant ('True'/'False')
                                                    2. faces list sorted by relevance (empty list if not relevant)
    Thus - the right use of this function is for example:
    - "if image_is_relevant.is_relevant:"
    - "for face in image_is_relevant(image).faces:"
    """
    Relevance = collections.namedtuple('relevance', 'is_relevant faces')
    faces = find_face(image, 10)
    return Relevance(len(faces) > 0, faces)


def find_face(image, max_num_of_faces=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        logging.warning("no good cascade found!")
        return []  # can we return [] in both cases or () in both , currently its one and on

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
        return faces  # can we return [] in both cases or () in both , currently its one and one
    return choose_faces(image, faces, max_num_of_faces)


def choose_faces(image, faces_list, max_num_of_faces):
    h, w, d = image.shape
    x_origin = int(w / 2)
    y_origin = int(0.125 * h)
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
    if 0.08 * image.shape[0] < h < 0.25 * image.shape[0] \
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
    bb_w = int((bb1[2] + bb2[2]) / 2)  # this isnt necessarily width, it could be x2 if rect is [x1,y1,x2,y2]
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
    x, y, w, h = face[0]
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
    mask[y_up:y_down, x_back:x_ahead] = 2
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


# can we move this to Utils or the like
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
    # image_counter = 0
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
        face = find_face(image)
        if len(face) > 0:                                # grabcut with mask
            rectangles = body_estimation(image, face)
            mask = create_mask_for_gc(rectangles, image)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
            # album.append(cv2.bitwise_and(image, image, mask=mask2))
            # image_counter += 1
        else:                                             # grabcut with rect
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
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
    x, y, w, h = face_rect[0]
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


def get_item_percentage(image_url, mask, labels):
    image = Utils.get_cv2_img_array(image_url)
    if image is None:
        logging.warning("image url is bad!")
        return None
    image = standard_resize(image, 400)[0]
    relevance = image_is_relevant(image)
    if not relevance.is_relevant:
        logging.warning("no face was found!")
        return None
    x, y, w, h = relevance.faces[0]
    face_size = w * h
    if 'dress' not in labels.keys():
        logging.warning("No dress was detected, good boy PD!")
        return None
    num_of_dress = labels.values()[labels.keys().index('dress')]
    dress_mask = np.where(mask == num_of_dress, 255, 0)
    num_dress_pix = cv2.countNonZero(dress_mask)
    return float(num_dress_pix) / face_size


images_urls = [
    'http://cdn.shopify.com/s/files/1/0115/5832/products/DIS043-AWholeNewWorld-RacerSkaterDress-6-WEB_large.jpg?v=1404964871',
    'http://cdn.shopify.com/s/files/1/0363/6989/products/PRI-UnEarthlyDelights-PlayDress-2-WEB_1024x1024.jpeg?v=1412221140',
    'http://cdn.shopify.com/s/files/1/0115/5832/products/ShatteredCrystalRevStrapSkaterDress-1-WEB_1024x1024.png?v=1391635312',
    'http://g-ecx.images-amazon.com/images/G/01/Shopbop/p/pcs/products/loves/loves2002213149/loves2002213149_q1_1-0_336x596.jpg',
    'http://www.maykool.com/media/catalog/product/m/u/multi-scoop-neck-frozen-print-casual-mini-dress-016280.jpg',
    'http://www.dylanqueen.co.uk/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/w/d/wdnbn00524.jpg',
    'http://www.kissydress.co.uk/images/product/beaded-chiffon-a-line-sweetheart-long-prom-dress-3183-6.jpg',
    'http://www.1dress.co.uk/media/catalog/product/cache/1/small_image/225x300/9df78eab33525d08d6e5fb8d27136e95/u/k/uk021581-2012-style-a-line-halter-beading-sleeveless-floor-length-chiffon-prom-dresses-evening-dresses-30627.jpg',
    'http://img.dressfirm.co/cdnimgs/dressfirm/other/ed15281-small-69%201.jpg',
    'http://demandware.edgesuite.net/sits_pod24/dw/image/v2/AAQB_PRD/on/demandware.static/-/Sites-whistles-master-catalog/default/dw9c47a5ed/images/00502074101/whistles-jersey-flippy-textured-dress-black_02.jpg?sw=319&sh=486&sm=fit&cx=750&cy=0&cw=980&ch=1494',
    'http://s7d5.scene7.com/is/image/Guess/W51GDGMP103-TWHM?$2014_G_large$',
    'http://img.davidsbridal.com/is/image/DavidsBridalInc/F17019_BM_PROD4_0227_FRONT_H_GUAVA?wid=346&qlt=100&resMode=sharp2&op_sharpen=1&bgc=255,255,255']

not_dress_urls = [
    'https://s-media-cache-ak0.pinimg.com/736x/38/0f/30/380f30b1340b9f4730bf8f8c62d9cdc8.jpg',
    'https://cdn.lookastic.com/looks/jacket-dress-shirt-capri-pants-heeled-sandals-sunglasses-large-9647.jpg',
    'https://cdn.lookastic.com/looks/crew-neck-t-shirt-dress-pants-pumps-sunglasses-bracelet-large-7655.jpg',
    'http://www.fashionistix.com/wp-content/uploads/2013/05/black-pants-white-shirt-for-women.jpg',
    'http://staff-jeans.com/wp-content/uploads/2015/03/01-7_i.jpg',
    'https://s-media-cache-ak0.pinimg.com/736x/59/e4/e7/59e4e732dfecb70eee3001325e0ea0e1.jpg',
    'https://s-media-cache-ak0.pinimg.com/736x/e8/ee/47/e8ee47fc34bfcd50e4b9b0e2422ebac9.jpg',
    'http://aelida.com/wp-content/uploads/2012/12/boho-maxi-skirt.jpg',
    'http://images-ff.asos-media.com/shmotterstorage/147711/large_a47d06df-0893-4347-b760-1e9511e5f393.jpeg',
    'https://i.imgur.com/U1LS5Kw.jpg',
    'http://www.sheplanet.com/wp-content/uploads/2012/05/Silver-Womens-Pencil-Skirt-520x780.jpg']


if __name__ == '__main__':
    print('starting')
