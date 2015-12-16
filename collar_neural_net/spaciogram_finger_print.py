
import numpy as np
import cv2


def spaciogram_finger_print(image, mask):
    '''
    :param image: cv2.BGR arrangement (numpy.array) - a must!
    :param mask: default is 0 / 255, but we check if else and fit for the inner workings.
    :return: spaciogram - a multi-dimensional histogram (numpy.array)
    '''

    ############ CHECKS ############
    # check if var|image is a numpy array of NxMx3:

    # checks if var|mask is a numpy array of NxM:

    # check if image and mask NxM are same:

    # check if mas is binary, 0/1, or 0/255:

    ############ CALCS ############
    # B channle:

    # G channle:

    # R channle:

    # R/G channle:

    # G/B channle:

    # B/R channle:

    # edge distance:

    # skeleton distance:

    # edges B channle:

    # edges G channle:

    # edges R channle:


def channles_of_BGR_image(image):
    '''
    :param image: cv2.BGR arrangement (numpy.array) - a must!
    :return: list of analysis images (list of numpy.array)
    '''
    normalized_image = []
    cv2.normalize(image, normalized_image, 0, 255, cv2.NORM_MINMAX)
    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]
    image_nB = normalized_image[:, :, 0]
    image_nG = normalized_image[:, :, 1]
    image_nR = normalized_image[:, :, 2]
    image_GRAY = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    image_nGRAY = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)

    image_B[image_B <= 0] = 1
    image_G[image_G <= 0] = 1
    image_R[image_R <= 0] = 1
    image_nB[image_nB <= 0] = 1
    image_nG[image_nG <= 0] = 1
    image_nR[image_nR <= 0] = 1
    image_GRAY[image_GRAY <= 0] = 1
    image_nGRAY[image_nGRAY <= 0] = 1

    image_BdG = image_B.astype(np.float16)/image_R.astype(np.float16)
    image_GdR = image_G.astype(np.float16)/image_R.astype(np.float16)
    image_RdB = image_G.astype(np.float16)/image_R.astype(np.float16)

    image_BdG = remap(image_BdG, np.amin(image_BdG), np.amax(image_BdG), 0, 255).astype(np.uint8)
    image_GdR = remap(image_GdR, np.amin(image_GdR), np.amax(image_GdR), 0, 255).astype(np.uint8)
    image_RdB = remap(image_RdB, np.amin(image_RdB), np.amax(image_RdB), 0, 255).astype(np.uint8)




def skeleton(blob_mask):
    '''
    :param blob_mask: default is 0 / 255, but we check if else and fit for the inner workings.
    :return skel: skeleton as by 0 / 255.
    '''

    size = np.size(blob_mask)
    skel = np.zeros(blob_mask.shape, np.uint8)
    ret, img = cv2.threshold(blob_mask, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel