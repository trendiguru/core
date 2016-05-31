
import numpy as np
import cv2

# ------------------ FINGERPRINTS FUNCTIONS --------------------- #

def fingerprint_3D_spatiogram(image, mask):
    bins = [8,12,8]
    # bins = [12, 12, 6]
    levelnum = 6
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    weights_circumfrance = circumfrence_distance(mask)
    gradient_mask0, gradient_mask1 = gradient_boxes(mask)

    circ_mask = spatial_mask_creator_from_gradient_mask(weights_circumfrance, levelnum, mask)
    vert_mask = spatial_mask_creator_from_gradient_mask(gradient_mask0, levelnum, mask)
    horz_mask = spatial_mask_creator_from_gradient_mask(gradient_mask1, levelnum, mask)

    masks = []
    for c in circ_mask:
        masks.append(np.array(255*c, np.uint8))
        # for a in vert_mask:
        #     for b in horz_mask:
        #         masks.append(np.array(255*a*b*c, np.uint8))
        #         # cv2.imshow('L', masks[-1])
        #         # cv2.waitKey(0)

    hist_list = []
    for mask in masks:
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
        hist_list.append(cv2.normalize(hist, hist).flatten())
    return np.array(hist_list)


def circumfrence_distance(blob_mask):
    '''
    :param blob_mask:
    :return:
    '''
    bounding_box = cv2.boundingRect(blob_mask)
    box = bounding_box[2:]
    kernel_size = np.sqrt(max(box)).astype('uint8')

    # kernel = np.ones((10, 10), np.uint8)
    # blob_mask = cv2.erode(blob_mask, kernel, iterations=1)

    # distance method:
    distance_method = cv2.DIST_C
    # DIST_C
    # DIST_L1
    # DIST_L2
    circumfrencen_distance_mask = cv2.distanceTransform(blob_mask, distance_method, 3)**0.67
    cv2.normalize(circumfrencen_distance_mask, circumfrencen_distance_mask, 0, 1., cv2.NORM_MINMAX)
    circumfrencen_distance_mask = np.asarray(255 * circumfrencen_distance_mask, np.uint8)
    dst = cv2.blur(circumfrencen_distance_mask, (kernel_size, kernel_size))
    circumfrencen_distance_mask = remap(dst, np.amin(dst), np.amax(dst), 0, 255)
    return circumfrencen_distance_mask


def gradient_boxes(mask):
    # direction is 0:horizontal and 1:vertical ;
    bounding_box = cv2.boundingRect(mask)
    bounding_box = [bounding_box[1], bounding_box[0], bounding_box[3], bounding_box[2]]
    box = bounding_box[2:]
    onesy = np.ones(box).astype('uint8')

    grad0 = np.arange(0., 256., 256. / box[0])
    if onesy.shape[0] != grad0.shape[0]:
        grad0 = grad0[:onesy.shape[0]]
        grad0 = remap(grad0, grad0.min(), grad0.max(), 0, 255)
    grad1 = np.arange(0., 256., 256. / box[1])
    if onesy.shape[1] != grad1.shape[0]:
        grad1 = grad1[:onesy.shape[1]]
        grad1 = remap(grad1, grad1.min(), grad1.max(), 0, 255)

    zero_mask = np.zeros(mask.shape).astype('uint8')
    gradient_mask0 = zero_mask
    gradient_mask1 = zero_mask

    grad0 = (onesy.T * grad0).T
    gradient_mask0[bounding_box[0]:bounding_box[0]+box[0], bounding_box[1]:bounding_box[1]+box[1]] = np.array(grad0).astype('uint8')
    temp0 = gradient_mask0.copy()
    grad1 = (onesy * grad1)
    gradient_mask1[bounding_box[0]:bounding_box[0]+box[0], bounding_box[1]:bounding_box[1]+box[1]] = np.array(grad1).astype('uint8')
    temp1 = gradient_mask1.copy()
    gradient_masks = [temp0, temp1]
    return gradient_masks


def spatial_mask_creator_from_gradient_mask(grad_mask, levelnum, global_mask):
    grad_mask[global_mask == 0] = 0
    max_level = grad_mask.max()
    min_level = grad_mask.min() + 1
    grad_mask = remap(grad_mask, min_level, max_level, 1, 255)
    # if min_level+1 != 0 or max_level != 255:
    #     grad_mask = remap(grad_mask, min_level, max_level, 1, 255)
    delta = np.arange(1., 256., 256. / levelnum)
    masks = []
    mask = np.zeros(global_mask.shape)
    for range_index in range(len(delta)):
        if range_index == 0:
            p1 = mask.copy()
            p1[grad_mask <= delta[1]] = 1
            p2 = mask.copy()
            p2[grad_mask > 0] = 1
            p3 = p1 * p2
            masks.append(p3)
        elif range_index < len(delta)-1 and range_index >= 1:
            p1 = mask.copy()
            p1[grad_mask <= delta[range_index+1]] = 1
            p2 = mask.copy()
            p2[grad_mask > delta[range_index]] = 1
            p3 = p1 * p2
            masks.append(p3)
        elif range_index == len(delta)-1:
            p1 = mask.copy()
            p1[grad_mask <= 256] = 1
            p2 = mask.copy()
            p2[grad_mask > delta[-1]] = 1
            p3 = p1 * p2
            masks.append(p3)
    return masks


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
    portion = (x-oldMin)*(float(newMax-newMin)/(oldMax-oldMin))
    if reverseInput:
        portion = (oldMax-x)*(float(newMax-newMin)/(oldMax-oldMin))

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion
    result = np.array(result).astype('uint8')
    return result

# ------------------ SEARCHING FUNCTIONS -------------------------- #

def chi2_distance(histA, histB):
    eps = 1e-10
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
    # return the chi-squared distance
    return d


def spatiogram_fingerprints_distance(query_fp, target_fp):
    '''
    :param spaciogram_1:
    :param spaciogram_2:
    :param filter_rank:
    :return:
    '''
    ############ CHECKS ############
    # check if spaciogram_1.shape == target_fp.shape:
    rating = []

    query_fp = np.array(query_fp, np.float32)
    target_fp = np.array(target_fp, np.float32)

    if query_fp.shape != target_fp.shape:
        # print 'Error: the dimensions of query_fp and target_fp are not equal! \n' \
        #       'shapes are: 1st - ' + str(np.array(query_fp).shape) + '\n' \
        #       'shapes are: 2nd - ' + str(np.array(target_fp).shape)
        return rating

    # HISTCMP_CORREL Correlation
    # HISTCMP_CHISQR Chi-Square
    # HISTCMP_INTERSECT Intersection
    # HISTCMP_BHATTACHARYYA Bhattacharyya distance
    # HISTCMP_HELLINGER Synonym for HISTCMP_BHATTACHARYYA
    # HISTCMP_CHISQR_ALT
    # HISTCMP_KL_DIV

    rating = []
    for i in range(np.array(query_fp).shape[0]):
        # rating.append(chi2_distance(query_fp[i], target_fp[i]))
        rating.append(cv2.compareHist(query_fp[i], target_fp[i], cv2.HISTCMP_BHATTACHARYYA))

    rating = np.array(rating).astype('float32')
    # rating = cv2.normalize(rating, rating)
    # print rating
    # rating = cv2.compareHist(rating, np.zeros(rating.shape[0]).astype('float32'), cv2.HISTCMP_CHISQR)
    # rating = chi2_distance(rating, np.zeros(len(rating)).astype('float32'))
    rating = rating.min()#np.sum(rating**2)#rating.max()#np.average(rating)#
    # print rating
    # rating = emd(query_fp, target_fp)
    return rating
