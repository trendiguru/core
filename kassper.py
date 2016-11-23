__author__ = 'Nadav Paz'

import numpy as np
import cv2
# from . import background_removal


def clutter_removal(image, thresh):     # non-recursive
    mask = get_mask(image)
    h = mask.shape[0]
    w = mask.shape[1]
    mask[0:h-1, 0] = 0
    mask[0:h-1, w-1] = 0
    mask[0, 0:w-1] = 0
    mask[h-1, 0:w-1] = 0

    def find_blob(pixel):    # update mask with 2's at removal candidates
        currblob = []
        potential_blob = []
        potential_blob.append(pixel)

        def find_relevant_neighbors(pixel):
            y = pixel[0]
            x = pixel[1]
            if mask[y][x-1] == 255:
                potential_blob.append((y, x-1))
            if mask[y-1][x] == 255:
                potential_blob.append((y-1, x))
            if mask[y][x+1] == 255:
                potential_blob.append((y, x+1))
            if mask[y+1][x] == 255:
                potential_blob.append((y+1, x))
            return

        for pixel in potential_blob:
            y = pixel[0]
            x = pixel[1]
            if mask[y][x] == 255:
                currblob.append(pixel)
                find_relevant_neighbors(pixel)
                mask[y][x] = 2
        return currblob

    for i in range(0, h):
        for j in range(0, w):
            if mask[i][j] == 255:
                currblob = find_blob((i, j))
                for (n, m) in currblob:
                    if len(currblob) < thresh:    # classified as clutter
                        mask[n][m] = 0
                    else:
                        mask[n][m] = 1
    return mask


def get_mask(image):
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 0:
                mask[i][j] = 0
            else:
                mask[i][j] = 255
    return mask


# def skin_removal(gc_image, image):
#     rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
#     bgdmodel = np.zeros((1, 65), np.float64)
#     fgdmodel = np.zeros((1, 65), np.float64)
#     ycrcb = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     face_rect = background_removal.find_face_cascade(image)
#     if len(face_rect) > 0:
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         x, y, w, h = face_rect[0]
#         face_image = image[y:y + h, x:x + w, :]
#         face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
#         bins = 180
#         n_pixels = face_image.shape[0] * face_image.shape[1]
#         hist_hue = cv2.calcHist([face_hsv], [0], None, [bins], [0, 180])
#         hist_hue = np.divide(hist_hue, n_pixels)
#         skin_hue_list = []
#         for l in range(0, 180):
#             if hist_hue[l] > 0.013:
#                 skin_hue_list.append(l)
#         for i in range(0, gc_image.shape[0]):
#             for j in range(0, gc_image.shape[1]):
#                 if hsv[i][j][0] in skin_hue_list and ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < \
#                         ycrcb[i][j][2] < 120:
#                     mask[i][j] = 2
#                 else:
#                     mask[i][j] = 3
#     else:
#         for i in range(0, gc_image.shape[0]):
#             for j in range(0, gc_image.shape[1]):
#                 if ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < ycrcb[i][j][2] < 120:
#                     mask[i][j] = 2
#                 else:
#                     mask[i][j] = 3
#     cv2.grabCut(gc_image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
#     mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
#     # without_skin = background_removal.get_masked_image(gc_image, mask2)
#     # return without_skin
#     return mask2


def skin_detection_with_grabcut(gc_image, image, face=None, skin_or_clothes='clothes'):
    rect = (0, 0, gc_image.shape[1] - 1, gc_image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    ycrcb = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
    partly_hsv = cv2.cvtColor(gc_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(gc_image.shape[:2], dtype=np.uint8)
    for i in range(0, gc_image.shape[0]):
        for j in range(0, gc_image.shape[1]):
            #skin thresholds: 80<=Cb<=120, 133<=Cr<=173 , from http://www.wseas.us/e-library/conferences/2011/Mexico/CEMATH/CEMATH-20.pdf
            if ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < ycrcb[i][j][2] < 120:
                if skin_or_clothes is 'clothes':
                    mask[i][j] = 2
                else:
                    mask[i][j] = 3
            else:
                if skin_or_clothes is 'clothes':
                    mask[i][j] = 3
                else:
                    mask[i][j] = 2
    if (mask == 2).all():
        return np.zeros(gc_image.shape[:2], dtype=np.uint8)
    else:
        cv2.grabCut(gc_image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0)
        # detected_image = background_removal.get_masked_image(gc_image, mask2)
        return mask2

def skin_detection(image_arr, face=None):
    '''
    return mask with skin as 255 and the rest 0
    todo - if a face is given use that to determine skintone
    :param image_arr:
    :param face:
    :return:
    '''
    ycrcb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2YCR_CB)
    mask = np.zeros(image_arr.shape[:2], dtype=np.uint8)
    for i in range(0, image_arr.shape[0]):
        for j in range(0, image_arr.shape[1]):
            #skin thresholds: 80<=Cb<=120, 133<=Cr<=173 , from http://www.wseas.us/e-library/conferences/2011/Mexico/CEMATH/CEMATH-20.pdf
            # Y>0 is added to those
            if ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < ycrcb[i][j][2] < 120:
                mask = 1
        return mask

# def create_item_mask(image):
#     """
#     this function will manage the isolation of the item and will create it's mask
#     :param image: 3d numpy array (BGR)
#     :return: 2d numpy array annotated mask with
#              0 = background
#              1 = skin
#              2 = the item
#     """
#     h, w = image.shape[:2]
#     mask = 2*np.ones((h, w), dtype=np.uint8)
#     h_margin = int(0.05*h)
#     w_margin = int(0.05*w)
#     mask[h_margin:h-h_margin, w_margin:w-w_margin] = 3
#     wo_bckgnd_mask = background_removal.simple_mask_grabcut(image, mask=mask)
#     skin_mask = skin_detection_with_grabcut(background_removal.get_masked_image(image, wo_bckgnd_mask.astype(np.uint8)),
#                                             image, skin_or_clothes='skin')
#     outmask = np.where(wo_bckgnd_mask == 255, 255, 0)
#     outmask = np.where(skin_mask == 255, 100, outmask)
#     return outmask