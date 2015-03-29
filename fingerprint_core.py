__author__ = 'liorsabag'

import numpy as np
import cv2
import string
import logging
import constants
"""
import classify_core
import Utils

DEFAULT_CLASSIFIERS = ["/home/www-data/web2py/applications/fingerPrint/modules/shirtClassifier.xml",
                       "/home/www-data/web2py/applications/fingerPrint/modules/pantsClassifier.xml",
                       "/home/www-data/web2py/applications/fingerPrint/modules/dressClassifier.xml"]
"""

fingerprint_length = constants.fingerprint_length

def find_color_percentages(img_array):
    """

    :param img_array:
    :return:bins for each color including black, white, gray
    """

    white_saturation_max = 36  # maximum S value for a white pixel14 from the gimp , 14*100/255
    white_value_min = 214 #  minimum V value for a white pixel this is 84 * 100/255
    black_value_max = 23 # maximum V value for a black pixel, 15*100/255
    n_colors=10
    color_limits=range(0,180+int(180/n_colors),int(180/n_colors))  #edges of bins for histogram
    #print(color_limits)

    h_arr=hsv[:,:,0]
    s_arr=hsv[:,:,1]
    v_arr=hsv[:,:,2]

   # mask[0] = image[0]==0
   # mask[1] = image[1]==0
   # mask[2] = image[2]==0
    #masksofi = mask[0] * mask[1] * mask[2]

    h, w, depth = img_array.shape
    area = h*w

    black_count=np.sum(v_arr<black_value_max)
    black_percentage=black_count/area

    #white is where saturation is less than sat_max and value>val_min
    white_mask=(s_arr<white_saturation_max) *(v_arr>white_value_min)
    white_count=np.sum(white_mask)
    white_percentage=white_count/area

    grey_count=np.sum((s_arr<white_saturation_max) *( v_arr<=white_value_min) *( v_arr>=black_value_max))
    grey_percentage=grey_count/area

    non_white=np.invert(white_mask)
    color_mask=non_white*(v_arr>=black_value_max)   #colors are where value>black, but not white
    colors_count=np.sum(color_mask)
    print("tot color count:"+str(tot_colors))
    color_counts=[]
    for i in range(0,n_colors):
        color_percentages.append(np.sum(  color_mask*(h_arr<color_limits[i+1])*(h_arr>=color_limits[i])))
        if DEBUG:
            print('color '+str(i)+' count ='+str(color_percentages[i]))
            print('color percentages:'+str(color_percentages))
        color_percentages[i]=color_percentages[i]/area  #turn pixel count into percentage
    all_colors=np.zeros(3)
    all_colors[0]=white_percentage
    all_colors[1]=black_percentage
    all_colors[2]=grey_percentage
    all_colors=np.append(all_colors,color_counts)

    #   all_colors=np.concatenate(all_colors,color_counts)
    if DEBUG:
        print('white black grey colors:'+str(all_colors))   #order is : white, black, grey, color_count[0]...color_count[n_colors]
        print('sum:'+str(np.sum(all_colors)))
 #   all_colors=color_counts
 #   np.append(all_colors,white_count)
 #   np.append(all_colors,black_count)
 #   all_colors.append(grey_count)

    #dominant_color_indices, dominant_colors = zip(*sorted(enumerate(all_colors), key=itemgetter(1), reverse=True))
    #above is for array, now working with numpy aray

# the order of dominant colors is what ccny guys used, if we just have vector in order of color i think its just as good
#so for now the following 3 lines are not used
    dominant_color_indices=np.argsort(all_colors, axis=-1, kind='quicksort', order=None)
    dominant_color_indices = dominant_color_indices[::-1]
    dominant_color_percentages=np.sort(all_colors, axis=-1, kind='quicksort', order=None)
    dominant_color_percentages = dominant_color_percentages[::-1]

    if DEBUG:
        print('color percentages:'+str(dominant_color_percentages)+' indices:'+str(dominant_color_indices))
    t1=time.time()
    return(all_colors)

def crop_image_to_bb(img, bb_coordinates_string_or_array):
    if isinstance(bb_coordinates_string_or_array, basestring):
        bb_array = [int(bb) for bb in string.split(bb_coordinates_string_or_array)]
    else:
        bb_array = bb_coordinates_string_or_array

    x = bb_array[0]
    y = bb_array[1]
    w = bb_array[2]
    h = bb_array[3]
    hh, ww, d = img.shape
    if (x + w <= ww) and (y + h <= hh):
	cropped_img = img[y:y+h,x:x+w]
    else:
        cropped_img = img
        logging.warning('Could not crop. Bad bounding box: imsize:' + str(ww) + ',' + str(hh) +
                        ' vs.:' + str(x + w) + ',' + str(y + h))

    return cropped_img

def fp(img, bounding_box=None, weights = np.ones(fingerprint_length) , histogram_length=25, use_intensity_histogram=False):
    if (bounding_box is not None) and (bounding_box != np.array([0, 0, 0, 0])).all():
        img = crop_image_to_bb(img, bounding_box)
    #crop out the outer 1/s of the image for color/texture-based features
    if img is None:
        return None
    s = 5
    h = img.shape[1]
    w = img.shape[0]

    if h==0 or w==0:
        return None

    r = [h / s, w / s, h - 2 * h / s, w - 2 * w / s]
#    roi = np.zeros((r[3], r[2], 3), np.uint8)
    roi = crop_image_to_bb(img,r)
    n_pixels = roi.shape[0] * roi.shape[1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    bins = histogram_length

    hist_hue = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  #flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], None, [bins], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], None, [bins], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  #flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    #Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    #Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  #this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps)/max_log_value
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps)/max_log_value
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps)/max_log_value
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    result_vector = np.concatenate((result_vector, hist_hue, hist_sat), axis=0)
    if use_intensity_histogram:
        result_vector = np.concatenate((result_vector,hist_int), axis=0)

    result_vector = result_vector * weights
    return result_vector


def my_range(start, stop, inc):
    r = start
    while r < stop:
        yield r
        r += inc

