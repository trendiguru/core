__author__ = 'Nadav Paz'

import numpy as np
import cv2
from trendi import background_removal


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
#this seems to have two probs, 1. loop over pixels in python and 2. return is within outer loop???
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
            if 30 < ycrcb[i][j][0] < 220  and 133 < ycrcb[i][j][1] < 173 and 80 < ycrcb[i][j][2] < 120:
                mask = 1
        n=np.count_nonzero(mask)
        print('skin pixels:'+str(n))
        return mask

def skin_detection_fast(image_arr, face=None):
    '''
    return mask with skin as 255 and the rest 0
    todo - if a face is given use that to determine skintone
    y -  cr [120...170]  cb[90..130]
    :param image_arr:
    :param face:
    :return:
    '''

    ff_cascade = background_removal.find_face_cascade(image_arr, max_num_of_faces=10)
    print('ffcascade:'+str(ff_cascade))
    if ff_cascade['are_faces'] :
        faces = ff_cascade['faces']
        if faces == []:
            print('ffascade reported faces but gave none')
        else:
            face = background_removal.choose_faces(faces,1)
            skin_colors = background_removal.face_skin_color_estimation_gmm(image_arr,face)



    ycrcb = cv2.cvtColor(image_arr, cv2.COLOR_BGR2YCR_CB)
#    mask = cv2.inRange(ycrcb,np.array([80,135,85]),np.array([255,180,135]))
    mask = cv2.inRange(ycrcb,np.array([90,140,95]),np.array([240,170,130]))
    # mask2 = cv2.inRange(ycrcb,np.array([0,0,0]),np.array([133,255,255]))
    # mask3 = cv2.inRange(ycrcb,np.array([0,0,0]),np.array([255,255,120]))
    # mask = mask1*mask2*mask3
    mask = np.where(mask  ==0,0,1).astype('uint8')  #return a 0,1 mask , easier for multiplication #
    n=np.count_nonzero(mask)
    print('skin pixels:'+str(n))
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
