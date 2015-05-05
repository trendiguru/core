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
    ycrcb = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
    skin_image = np.zeros(gc_image.shape, np.uint8)
    clothes_image = np.zeros(gc_image.shape, np.uint8)
    face_rect = background_removal.find_face(image)
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
                    for k in range(0, 3):
                        skin_image[i][j][k] = gc_image[i][j][k]
                else:
                    for k in range(0, 3):
                        clothes_image[i][j][k] = gc_image[i][j][k]
    else:
        for i in range(0, gc_image.shape[0]):
            for j in range(0, gc_image.shape[1]):
                if ycrcb[i][j][0] > 0 and 133 < ycrcb[i][j][1] < 173 and 80 < ycrcb[i][j][2] < 120:
                    for k in range(0, 3):
                        skin_image[i][j][k] = gc_image[i][j][k]
                else:
                    for k in range(0, 3):
                        clothes_image[i][j][k] = gc_image[i][j][k]
    return clothes_image