
import numpy as np
import cv2

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
    # B, G, R, B/G, G/R, R/B ,edge distance, skeleton distance, channels:

    bins = 6

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
    channels_list = channles_of_BGR_image(image)
    skell_dist  = skeleton_distance(mask)
    circ_dist = circumfrence_distance(mask)
    sample = []
    for channel in channels_list:
        sample.append(channel[mask>0].flatten())
    sample.append(skell_dist[mask>0].flatten())
    sample.append(circ_dist[mask>0].flatten())
    spaciogram, edges = np.histogramdd(sample, bins, normed=True, weights=None)
    spaciogram = spaciogram.flatten()
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
    # B, G, R, B/G, G/R, R/B ,edge distance, skeleton distance, channels:

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
    channels_list = channles_of_BGR_image(image)
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
    print spaciogram
    print spaciogram.shape
    return spaciogram

def channles_of_BGR_image(image):
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

    image_B[image_B <= 0] = 1
    image_G[image_G <= 0] = 1
    image_R[image_R <= 0] = 1
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

# image = cv2.imread('/home/nate/Desktop/wild_square_1.jpg')
# blob_mask = np.zeros(image.shape[:2], dtype=np.uint8)
# blob_mask[50:250, 50:250] = 255
# spaciogram_finger_print(image, blob_mask)