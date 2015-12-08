__author__ = 'Nadav Paz'

import numpy as np
import cv2

import background_removal


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


def skin_removal(gc_image, image):
    rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    ycrcb = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    face_rect = background_removal.find_face_cascade(image)
    if len(face_rect) > 0:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
        for i in range(0, gc_image.shape[0]):
            for j in range(0, gc_image.shape[1]):
                if hsv[i][j][0] in skin_hue_list and ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < \
                        ycrcb[i][j][2] < 120:
                    mask[i][j] = 2
                else:
                    mask[i][j] = 3
    else:
        for i in range(0, gc_image.shape[0]):
            for j in range(0, gc_image.shape[1]):
                if ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < ycrcb[i][j][2] < 120:
                    mask[i][j] = 2
                else:
                    mask[i][j] = 3
    cv2.grabCut(gc_image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    # without_skin = background_removal.get_masked_image(gc_image, mask2)
    # return without_skin
    return mask2


def skin_detection_with_grabcut(gc_image, image, face=None, skin_or_clothes='clothes'):
    rect = (0, 0, gc_image.shape[1] - 1, gc_image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    ycrcb = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
    partly_hsv = cv2.cvtColor(gc_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(gc_image.shape[:2], dtype=np.uint8)
    if len(face) == 0:
        face_rect = background_removal.find_face_cascade(image)['faces']
    if len(face_rect) > 0:
        skin_hue_list = background_removal.face_skin_color_estimation(image, face_rect)
        for i in range(0, gc_image.shape[0]):
            for j in range(0, gc_image.shape[1]):
                if partly_hsv[i][j][0] in skin_hue_list and ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < \
                        ycrcb[i][j][2] < 120:
                    if skin_or_clothes is 'clothes':
                        mask[i][j] = 2
                    else:
                        mask[i][j] = 3
                else:
                    if skin_or_clothes is 'clothes':
                        mask[i][j] = 3
                    else:
                        mask[i][j] = 2
    else:
        for i in range(0, gc_image.shape[0]):
            for j in range(0, gc_image.shape[1]):
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
    cv2.grabCut(gc_image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    detected_image = background_removal.get_masked_image(gc_image, mask2)
    return detected_image