import time

import numpy as np
import cv2

from emd import emd


def spaciograms_distance_rating(spaciogram_1, spaciogram_2):
    '''
    :param spaciogram_1:
    :param spaciogram_2:
    :return:
    '''
    ############ CHECKS ############
    # check if spaciogram_1.shape == spaciogram_2.shape:
    rating = []
    if np.array(spaciogram_1.shape).all() == np.array(spaciogram_2.shape).all() is False:
        print 'Error: the dimensions of spaciogram_1 and spaciogram_2 are not equal! \n' \
              'shapes are: 1st - ' + str(spaciogram_1.shape) + '\n' \
              'shapes are: 2nd - ' + str(spaciogram_2.shape)
        return rating

    # # Define number of rows (overall bin count):
    # numRows = spaciogram_1.size
    # dims = len(spaciogram_1.shape)
    # bins_per_dim = len(spaciogram_1)
    # signature_1 = np.zeros([numRows, dims+1]) #cv2.CreateMat(numRows, dims, cv2.CV_32FC1)
    # print signature_1.shape
    # signature_2 = signature_1 #cv2.CreateMat(numRows, dims, cv2.CV_32FC1)
    # sigrature_index = 0
    # # fill signature_natures:
    # # TODO: for production optimize this, use Numpy (reshape?)
    # for d1 in range(0, bins_per_dim - 1):
    #     for d2 in range(0, bins_per_dim - 1):
    #         for d3 in range(0, bins_per_dim - 1):
    #             for d4 in range(0, bins_per_dim - 1):
    #                 for d5 in range(0, bins_per_dim - 1):
    #                     # signature 1:
    #                     signature_1[sigrature_index, :] = [spaciogram_1[d1, d2, d3, d4, d5], d1, d2, d3, d4, d5]
    #                     # bin_val = cv2.QueryHistValue_2D(spaciogram_1, d1, d2, d3, d4, d5)
    #                     # cv.Set2D(signature_1, sigrature_index, 0, bin_val) #bin value
    #                     # cv.Set2D(signature_1, sigrature_index, 1, d1)  #coord1
    #                     # cv.Set2D(signature_1, sigrature_index, 2, d2) #coord2
    #                     # cv.Set2D(signature_1, sigrature_index, 3, d3)  #coord3
    #                     # cv.Set2D(signature_1, sigrature_index, 4, d4) #coord4
    #                     # cv.Set2D(signature_1, sigrature_index, 5, d5)  #coord5
    #                     # signature 2:
    #                     signature_2[sigrature_index, :] = [spaciogram_2[d1, d2, d3, d4, d5], d1, d2, d3, d4, d5]
    #                     # bin_val2 = cv2.QueryHistValue_2D(spaciogram_2, d1, d2, d3, d4, d5)
    #                     # cv.Set2D(signature_2, sigrature_index, 0, bin_val2) #bin value
    #                     # cv.Set2D(signature_2, sigrature_index, 1, d1)  #coord1
    #                     # cv.Set2D(signature_2, sigrature_index, 2, d2) #coord2
    #                     # cv.Set2D(signature_2, sigrature_index, 3, d3)  #coord3
    #                     # cv.Set2D(signature_2, sigrature_index, 4, d4) #coord4
    #                     # cv.Set2D(signature_2, sigrature_index, 5, d5)  #coord5
    #                     sigrature_index += 1
    #                     print spaciogram_1[d1, d2, d3, d4, d5]

    signature_1 = np.zeros([spaciogram_1.size / len(spaciogram_1),  len(spaciogram_1)])
    sigrature_index = 0
    # print len(spaciogram_1)
    for dim in spaciogram_1:
        signature_1[:, sigrature_index] = dim.flatten()
        sigrature_index += 1

    signature_2 = np.zeros([spaciogram_2.size / len(spaciogram_1),  len(spaciogram_2)])
    sigrature_index = 0
    for dim in spaciogram_2:
        signature_2[:, sigrature_index] = dim.flatten()
        sigrature_index += 1

    # signature_1 = np.reshape(spaciogram_1, (spaciogram_1[0].size, len(spaciogram_1)))
    # signature_2 = np.reshape(spaciogram_2, (spaciogram_2[0].size, len(spaciogram_2)))

    method = cv2.HISTCMP_BHATTACHARYYA
    # HISTCMP_CORREL Correlation
    # HISTCMP_CHISQR Chi-Square
    # HISTCMP_INTERSECT Intersection
    # HISTCMP_BHATTACHARYYA Bhattacharyya distance
    # HISTCMP_HELLINGER Synonym for HISTCMP_BHATTACHARYYA
    # HISTCMP_CHISQR_ALT
    # HISTCMP_KL_DIV

    rating = cv2.compareHist(spaciogram_1.astype('float32'), spaciogram_2.astype('float32'), method)
    # rating = emd(signature_1, signature_2)
    return rating

def spaciogram_finger_print(image, mask):
    '''
    :param image: cv2.BGR arrangement (numpy.array) - a must!
    :param mask: default is 0 / 255, but we check if else and fit for the inner workings.
    :return: spaciogram - a multi-dimensional histogram  - flatten (numpy.array)
    '''

    ############ CHECKS ############
    # check if var|image is a numpy array of NxMx3:

    # checks if var|mask is a numpy array of NxM:

    # check if image and mask NxM are same:

    # check if mas is binary, 0/1, or 0/255:

    ############ CALCS ############
    # color channels ,edge distance, skeleton distance, channels:

    bins = 5

    # limiting the image size for a quicker calculation:
    limit = [500, 500]
    resize_interpulation = cv2.INTER_NEAREST#INTER_LINEAR#INTER_CUBIC#INTER_LANCZOS4#INTER_AREA#
    if image.shape[0] > limit[0] or image.shape[1] > limit[1]:
        delta = [1.0 * limit[0] / image.shape[0], 1.0 * limit[1] / image.shape[1]]
        resize_factor = min(delta)
        newx, newy = image.shape[1] * resize_factor, image.shape[0] * resize_factor
        image = cv2.resize(image, (int(newx), int(newy)), interpolation=resize_interpulation)
        mask = cv2.resize(mask, (int(newx), int(newy)), interpolation=resize_interpulation)

    # changing to an exact eucledian space model of color:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels_list = channles_of_image(image)
    skell_dist  = skeleton_distance(mask)
    circ_dist = circumfrence_distance(mask)

    # # opencv way:
    # channels_list.append(skell_dist)
    # channels_list.append(circ_dist)
    # spaciogram = cv2.calcHist(channels_list, range(len(channels_list)),
    #                           mask, bins, [0, 256])# * np.ones([1, len(channels_list)]), )

    sample = []
    for channel in channels_list:
        sample.append(channel[mask>0].flatten())
    sample.append(skell_dist[mask>0].flatten())
    sample.append(circ_dist[mask>0].flatten())
    spaciogram, edges = np.histogramdd(sample, bins, normed=True, weights=None)

    # spaciogram = spaciogram * (10**bins) #spaciogram.flatten()
    # print np.amax(spaciogram)
    return spaciogram

def histogram_stack_finger_print(image, mask):
    '''
    :param image: cv2.BGR arrangement (numpy.array) - a must!
    :param mask: default is 0 / 255, but we check if else and fit for the inner workings.
    :return: spaciogram - a 2DxN stack of histograms - flatten (numpy.array)
    '''

    ############ CHECKS ############
    # check if var|image is a numpy array of NxMx3:

    # checks if var|mask is a numpy array of NxM:

    # check if image and mask NxM are same:

    # check if mas is binary, 0/1, or 0/255:

    ############ CALCS ############
    # color channels ,edge distance, skeleton distance, channels:

    bins = 10

    # limiting the image size for a quicker calculation:
    limit = [1000, 1000]
    resize_interpulation = cv2.INTER_NEAREST#INTER_LINEAR#INTER_CUBIC#INTER_LANCZOS4#INTER_AREA#
    if image.shape[0] > limit[0] or image.shape[1] > limit[1]:
        delta = [1.0 * limit[0] / image.shape[0], 1.0 * limit[1] / image.shape[1]]
        resize_factor = min(delta)
        newx, newy = image.shape[1] * resize_factor, image.shape[0] * resize_factor
        image = cv2.resize(image, (int(newx), int(newy)), interpolation=resize_interpulation)
        mask = cv2.resize(mask, (int(newx), int(newy)), interpolation=resize_interpulation)

    # changing to an exact eucledian space model of color:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels_list = channles_of_image(image)
    skell_dist  = skeleton_distance(mask)
    circ_dist = circumfrence_distance(mask)
    sample = []
    for channel in channels_list:
        sample.append(channel[mask>0].flatten())

    skell_sample = skell_dist[mask>0].flatten()
    circ_sample = circ_dist[mask>0].flatten()
    spaciogram = []
    for channel in sample:
        skell_spaciogram, xedges, yedges = np.histogram2d(skell_sample, channel, bins, normed=True, weights=None)
        circ_spaciogram, xedges, yedges = np.histogram2d(circ_sample, channel, bins, normed=True, weights=None)
        spaciogram.append(np.hstack([skell_spaciogram.flatten(), circ_spaciogram.flatten()]))
    spaciogram = np.concatenate(spaciogram, axis=0)
    # print spaciogram
    return spaciogram

def channles_of_image(image):
    '''
    :param image: cv2.BGR arrangement (numpy.array) - a must!
    :return image_listing: list of analysis images (list of numpy.array)
    '''
    normalized_image = image
    # cv2.normalize(image, normalized_image, 0, 255, cv2.NORM_MINMAX)
    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]
    # image_nB = normalized_image[:, :, 0]
    # image_nG = normalized_image[:, :, 1]
    # image_nR = normalized_image[:, :, 2]
    # image_GRAY = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    # image_nGRAY = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)

    # image_B[image_B <= 0] = 1
    # image_G[image_G <= 0] = 1
    # image_R[image_R <= 0] = 1
    # image_nB[image_nB <= 0] = 1
    # image_nG[image_nG <= 0] = 1
    # image_nR[image_nR <= 0] = 1
    # image_GRAY[image_GRAY <= 0] = 1
    # image_nGRAY[image_nGRAY <= 0] = 1

    # image_BdG = image_B.astype(np.float16)/image_R.astype(np.float16)
    # image_GdR = image_G.astype(np.float16)/image_R.astype(np.float16)
    # image_RdB = image_G.astype(np.float16)/image_R.astype(np.float16)
    #
    # image_BdG = remap(image_BdG, np.amin(image_BdG), np.amax(image_BdG), 0, 255).astype(np.uint8)
    # image_GdR = remap(image_GdR, np.amin(image_GdR), np.amax(image_GdR), 0, 255).astype(np.uint8)
    # image_RdB = remap(image_RdB, np.amin(image_RdB), np.amax(image_RdB), 0, 255).astype(np.uint8)

    # image_listing = [image_B, image_G, image_R, image_nB, image_nG, image_nR,
    #                  image_GRAY, image_nGRAY, image_BdG, image_GdR, image_RdB]
    image_listing = [image_B, image_G, image_R]
    return image_listing

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

def skeleton_distance(blob_mask):
    '''
    :param blob_mask:
    :return:
    '''
    skeleton_mask = skeleton(blob_mask)
    skeleton_mask[skeleton_mask == 0] = 1
    skeleton_mask[skeleton_mask > 1] = 0
    skeleton_mask[skeleton_mask == 1] = 255
    skeleton_distance_mask = cv2.distanceTransform(skeleton_mask, cv2.DIST_L2, 3)
    cv2.normalize(skeleton_distance_mask, skeleton_distance_mask, 0, 1., cv2.NORM_MINMAX)
    return skeleton_distance_mask

def circumfrence_distance(blob_mask):
    '''
    :param blob_mask:
    :return:
    '''
    circumfrencen_distance_mask = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 3)
    cv2.normalize(circumfrencen_distance_mask, circumfrencen_distance_mask, 0, 1., cv2.NORM_MINMAX)
    return circumfrencen_distance_mask

def remap(x, oMin, oMax, nMin, nMax):

    #range check
    if oMin == oMax:
        print "Warning: Zero input range"
        return None

    if nMin == nMax:
        print "Warning: Zero output range"
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min(oMin, oMax)
    oldMax = max(oMin, oMax)
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False
    newMin = min(nMin, nMax)
    newMax = max(nMin, nMax)
    if not newMin == nMin :
        reverseOutput = True

# new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result