__author__ = 'liorsabag'
# TODO  consider using histogram - match metrics from here
# http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html#explanation

import string
import logging

import numpy as np
import cv2

import background_removal
import constants
import utils_tg








# moving this into the show_fp function for now - LS
# import matplotlib.pyplot as plt

fingerprint_length = constants.fingerprint_length
histograms_length = constants.histograms_length


def find_color_percentages(img_array):
    """

    :param img_array:
    :return:bins for each color including black, white, gray
    """

    white_saturation_max = 36  # maximum S value for a white pixel14 from the gimp , 14*100/255
    white_value_min = 214  # minimum V value for a white pixel this is 84 * 100/255
    black_value_max = 23  # maximum V value for a black pixel, 15*100/255
    n_colors = 10
    color_limits = range(0, 180 + int(180 / n_colors), int(180 / n_colors))  # edges of bins for histogram
    # print(color_limits)

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    h_arr = hsv[:, :, 0]
    s_arr = hsv[:, :, 1]
    v_arr = hsv[:, :, 2]

    # mask[0] = image[0]==0
    # mask[1] = image[1]==0
    # mask[2] = image[2]==0
    # masksofi = mask[0] * mask[1] * mask[2]

    h, w, depth = img_array.shape
    area = h * w

    black_count = np.sum(v_arr < black_value_max)
    black_percentage = float(black_count) / area

    # white is where saturation is less than sat_max and value>val_min
    white_mask = (s_arr < white_saturation_max) * (v_arr > white_value_min)
    white_count = np.sum(white_mask)
    white_percentage = float(white_count) / area

    grey_count = np.sum((s_arr < white_saturation_max) * (v_arr <= white_value_min) * (v_arr >= black_value_max))
    grey_percentage = float(grey_count) / area

    non_white = np.invert(white_mask)
    color_mask = non_white * (v_arr >= black_value_max)  # colors are where value>black, but not white
    colors_count = np.sum(color_mask)
    # print("tot color count:"+str(tot_colors))
    color_counts = []
    color_percentages = []
    for i in range(0, n_colors):
        color_percentages.append(np.sum(color_mask * (h_arr < color_limits[i + 1]) * (h_arr >= color_limits[i])))
        # print('color ' + str(i) + ' count =' + str(color_percentages[i]))
        color_percentages[i] = float(color_percentages[i]) / area  # turn pixel count into percentage
        #       print('color percentages:' + str(color_percentages))
    all_colors = np.zeros(3)
    all_colors[0] = white_percentage
    all_colors[1] = black_percentage
    all_colors[2] = grey_percentage

    #    all_colors=np.append(all_colors,color_percentages)

    #   all_colors=np.concatenate(all_colors,color_counts)

    # print('white black grey colors:' + str(all_colors))  # order is : white, black, grey, color_count[0]...color_count[n_colors]
    # print('sum:' + str(np.sum(all_colors)))
    #   all_colors=color_counts
    #   np.append(all_colors,white_count)
    #   np.append(all_colors,black_count)
    #   all_colors.append(grey_count)

    # dominant_color_indices, dominant_colors = zip(*sorted(enumerate(all_colors), key=itemgetter(1), reverse=True))
    # above is for array, now working with numpy aray

    # the order of dominant colors is what ccny guys used, if we just have vector in order of color i think its just as good
    # so for now the following 3 lines are not used
    dominant_color_indices = np.argsort(all_colors, axis=-1,
                                        kind='quicksort')  # make sure this is ok   TODO - took out order=None argument
    dominant_color_indices = dominant_color_indices[::-1]
    dominant_color_percentages = np.sort(all_colors, axis=-1,
                                         kind='quicksort')  # make sure this is ok   TODO - took out order=None argument
    dominant_color_percentages = dominant_color_percentages[::-1]

    #    print('color percentages:' + str(dominant_color_percentages) + ' indices:' + str(dominant_color_indices))
    return (all_colors)


def crop_image_to_bb(img, bb_coordinates_string_or_array):
    if isinstance(bb_coordinates_string_or_array, basestring):
        bb_array = [int(bb) for bb in string.split(bb_coordinates_string_or_array)]
    else:
        bb_array = bb_coordinates_string_or_array

    x, y, w, h = bb_array
    hh, ww, d = img.shape  # i think this will fail on grayscale imgs
    if (x + w <= ww) and (y + h <= hh):
        cropped_img = img[y:y + h, x:x + w]
    else:
        cropped_img = img
        logging.warning('Could not crop. Bad bounding box: imsize:' + str(ww) + ',' + str(hh) +
                        ' vs.:' + str(x + w) + ',' + str(y + h))
        person = input('Enter your name: ')

    return cropped_img


def fp(img, mask=None, weights=np.ones(fingerprint_length), histogram_length=25):
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print('trouble with mask size, resetting to image size')
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    n_pixels = cv2.countNonZero(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    # histograms
    bins = histogram_length

    hist_hue = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], mask, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], mask, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    # Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    # Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  # this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps) / max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps) / max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps) / max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)


    # weights = np.ones(len(result_vector))  # THIS IS A KLUGE , FIX
    result_vector = np.multiply(result_vector, weights)
    return result_vector


def fp_HSCrCb(img, mask=None, weights=np.ones(fingerprint_length), histogram_length=25):
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print('trouble with mask size, resetting to image size')
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    n_pixels = cv2.countNonZero(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    # histograms
    bins = histogram_length

    hist_Cr = cv2.calcHist([YCrCb], [1], mask, [bins], [0, 255])
    hist_Cr = [item for sublist in hist_Cr for item in sublist]  # flatten nested
    hist_Cr = np.divide(hist_Cr, n_pixels)

    hist_Cb = cv2.calcHist([YCrCb], [2], mask, [bins], [0, 255])
    hist_Cb = [item for sublist in hist_Cb for item in sublist]  # flatten nested
    hist_Cb = np.divide(hist_Cb, n_pixels)

    hist_hue = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], mask, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], mask, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    # Uniformity  t(5)=sum(p.^ 2);

    Cr_uniformity = np.dot(hist_Cr, hist_Cr)
    Cb_uniformity = np.dot(hist_Cb, hist_Cb)
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    # Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  # this is same as sum of p log p
    l_Cr = -np.log2(hist_Cr + eps) / max_log_value
    Cr_entropy = np.dot(hist_Cr, l_Cr)
    l_Cb = -np.log2(hist_Cb + eps) / max_log_value
    Cb_entropy = np.dot(hist_Cb, l_Cb)
    l_hue = -np.log2(hist_hue + eps) / max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps) / max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps) / max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [Cr_uniformity, Cb_uniformity, hue_uniformity, sat_uniformity, int_uniformity, Cr_entropy,
                     Cb_entropy, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_Cr, hist_Cb, hist_hue, hist_sat), axis=0)

    # weights = np.ones(len(result_vector))  # THIS IS A KLUGE , FIX
    # result_vector = np.multiply(result_vector, weights)
    return result_vector


def gc_and_fp_with_kwargs(img, bounding_box=None, weights=np.ones(fingerprint_length), **kwargs):
    # for kw in kwargs:
    # print('in fp_with_kwargs, kw:'+str(kw)+'='+str(kwargs[kw]))
    if 'histogram_length' in kwargs:
        histogram_length = kwargs['histogram_length']
    else:
        histogram_length = constants.histograms_length
    if bounding_box == None:
        print('warning - bad bounding box caught in gc_and_fp')
        bounding_box = [0, 0, img.shape[1], img.shape[0]]

    mask = background_removal.get_fg_mask(img, bounding_box=bounding_box)
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print('trouble with mask size, resetting to image size')
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    n_pixels = cv2.countNonZero(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    # histograms
    bins = histogram_length

    hist_hue = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], mask, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], mask, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    # Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    # Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  # this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps) / max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps) / max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps) / max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)

    weights = np.ones(len(result_vector))  # THIS IS A KLUGE , FIX
    result_vector = np.multiply(result_vector, weights)
    return result_vector


def fp_with_bwg(img, mask=None, weights=np.ones(fingerprint_length), histogram_length=25,
                **kwargs):  # with black, white, gray
    if mask is None or cv2.countNonZero(mask) == 0:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        print('trouble with mask size, resetting to image size')
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    n_pixels = cv2.countNonZero(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    # histograms
    bins = histogram_length

    hist_hue = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], mask, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], mask, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    # Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    # Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  # this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps) / max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps) / max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps) / max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)

    black_white_gray_percentages = find_color_percentages(img)
    # print('bwg:' + str(black_white_gray_percentages))
    result_vector = np.concatenate((result_vector, black_white_gray_percentages))

    # weights = np.ones(len(result_vector))
    if len(weights) != len(result_vector):
        logging.warning('len(wieghts)=' + str(len(weights)) + '!=len(fp):' + str(len(result_vector)))
        return result_vector
    result_vector = np.multiply(result_vector, weights)
    return result_vector


def gc_and_fp(img, bounding_box=None, weights=np.ones(fingerprint_length), histogram_length=25, **kwargs):
    if bounding_box == None:
        print('warning - bad bounding box caught in gc_and_fp')
        bounding_box = [0, 0, img.shape[1], img.shape[0]]

    mask = background_removal.get_fg_mask(img, bounding_box=bounding_box)
    fingerprint = fp(img, mask, weights=weights, **kwargs)
    return fingerprint


def gc_and_fp_YCrCb(img, bounding_box=None, weights=np.ones(fingerprint_length), histogram_length=25, **kwargs):
    if bounding_box == None:
        print('warning - bad bounding box caught in gc_and_fp')
        bounding_box = [0, 0, img.shape[1], img.shape[0]]

    mask = background_removal.get_fg_mask(img, bounding_box=bounding_box)
    fingerprint = fp_HSCrCb(img, mask, weights=weights, **kwargs)
    return fingerprint


def gc_and_fp_histeq(img, bounding_box=None, weights=np.ones(fingerprint_length), histogram_length=25, **kwargs):
    if bounding_box == None:
        print('warning - bad bounding box caught in gc_and_fp')
        bounding_box = [0, 0, img.shape[1], img.shape[0]]

    mask = background_removal.get_fg_mask(img, bounding_box=bounding_box)
    # . YCbCr is preferred as it is designed for digital images. Perform HE of the intensity plane Y. Convert the image back to RGB.
    equalized = eq_BGR(img)
    fingerprint = fp(equalized, mask, weights=weights, **kwargs)
    return fingerprint


def eq_BGR(img):
    # img2[:, :, 0] = cv2.equalizeHist(img2[:, :, 0])

    YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    Y_component = YCbCr[:, :, 0]
    Y_eq = cv2.equalizeHist(Y_component)
    YCbCr[:, :, 0] = Y_eq
    bgr = cv2.cvtColor(YCbCr, cv2.COLOR_YCR_CB2BGR)
    return (bgr)


def gc_and_fp_bw(img, bounding_box=None, weights=np.ones(fingerprint_length), **kwargs):
    if bounding_box == None:
        print('warning - bad bounding box caught in gc_and_fp')
        bounding_box = [0, 0, img.shape[1], img.shape[0]]

    mask = background_removal.get_fg_mask(img, bounding_box=bounding_box)
    fingerprint = fp_with_bwg(img, mask, weights=weights, **kwargs)
    return fingerprint


def regular_fp(img, bounding_box=None, weights=np.ones(fingerprint_length), **kwargs):
    if bounding_box == None:
        print('warning - bad bounding box caught in regular_fp')
        bounding_box = [0, 0, img.shape[1], img.shape[0]]
    mask = utils_tg.bb_to_mask(bounding_box, img)
    fingerprint = fp(img, mask, weights=weights)
    return fingerprint


def show_fp(fingerprint, fig=None, **kwargs):
    import matplotlib.pyplot as plt

    if fig:
        plt.close(fig)
    plt.close('all')

    energy_maxindex = constants.extras_length
    if 'histogram_length' in kwargs:
        histograms_length = kwargs['histogram_length']
    else:
        histograms_length = constants.histograms_length
    hue_maxindex = energy_maxindex + histograms_length
    sat_maxindex = hue_maxindex + histograms_length

    fig, ax = plt.subplots()
    ind = np.arange(len(fingerprint))  # the x locations for the groups
    width = 0.35

    rects1 = ax.bar(ind[0:energy_maxindex], fingerprint[0:energy_maxindex], width, color='r')  # , yerr=menStd)
    rects2 = ax.bar(ind[energy_maxindex: hue_maxindex], fingerprint[energy_maxindex: hue_maxindex], width,
                    color='g')  # , yerr=menStd)
    rects3 = ax.bar(ind[hue_maxindex: sat_maxindex], fingerprint[hue_maxindex: sat_maxindex], width,
                    color='b')  # , yerr=menStd)

    print('len fp' + str(len(fingerprint)) + ' sat_index:' + str(sat_maxindex))
    if len(fingerprint) > sat_maxindex:
        # do whatever is left
        extra = len(fingerprint) - sat_maxindex
        rects4 = ax.bar(ind[sat_maxindex:], fingerprint[sat_maxindex:], width, color='y')  # , yerr=menStd)

    # add some text for labels, title and axes tisatcks
    ax.set_ylabel('y')
    ax.set_title('fingerprint')
    ax.set_xticks(ind + width)
    # ax.set_xticklabels(rotation=45)
    #    ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )
    # ax.legend( (rects1[0]), ('Men', 'Women') )
    # raw_input('before')
    plt.show(block=True)
    # raw_input('after')
    return (fig)


def my_range(start, stop, inc):
    r = start
    while r < stop:
        yield r
        r += inc
