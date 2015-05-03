__author__ = 'Nadav Paz'

import numpy as np
import cv2


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


def skin_removal(image):
    YCrCb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    clothes_image = np.zeros((YCrCb_image.shape[0], YCrCb_image.shape[1], 3), np.uint8)
    for i in range(0, YCrCb_image.shape[0]):
        for j in range(0, YCrCb_image.shape[1]):
            if not 133 < YCrCb_image[i][j][1] < 173 and 80 < YCrCb_image[i][j][2] < 120:
                for k in range(0, 3):
                    clothes_image[i][j][k] = image[i][j][k]
    return clothes_image
