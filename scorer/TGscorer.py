import numpy as np
import cv2
import time
import os
import io
import json
import urllib
from joblib import Parallel, delayed
import multiprocessing


# from .. import find_similar_mongo
# from .. import background_removal
# from .. import constants
# from ..paperdoll import paperdoll_parse_enqueue
# from .. import paperdolls

original_paperdoll_weights_dictionary = {'background': 1,
                                         'blazer': 1,
                                         'cape': 1,
                                         'flats': 1,
                                         'jacket': 1,
                                         'pants': 1,
                                         'scarf': 1,
                                         'socks': 1,
                                         't-shirt': 1,
                                         'watch': 1,
                                         'skin': 1,
                                         'blouse': 1,
                                         'cardigan': 1,
                                         'glasses': 1,
                                         'jeans': 1,
                                         'pumps': 1,
                                         'shirt': 1,
                                         'stockings': 1,
                                         'tie': 1,
                                         'wedges': 1,
                                         'hair': 1,
                                         'bodysuit': 1,
                                         'clogs': 1,
                                         'gloves': 1,
                                         'jumper': 1,
                                         'purse': 1,
                                         'shoes': 1,
                                         'suit': 1,
                                         'tights': 1,
                                         'accessories': 1,
                                         'boots': 1,
                                         'coat': 1,
                                         'hat': 1,
                                         'leggings': 1,
                                         'ring': 1,
                                         'shorts': 1,
                                         'sunglasses': 1,
                                         'top': 1,
                                         'bag': 1,
                                         'bra': 1,
                                         'dress': 1,
                                         'heels': 1,
                                         'loafers': 1,
                                         'romper': 1,
                                         'skirt': 1,
                                         'sweater': 1,
                                         'vest': 1,
                                         'belt': 1,
                                         'bracelet': 1,
                                         'earrings': 1,
                                         'intimate': 1,
                                         'necklace': 1,
                                         'sandals': 1,
                                         'sneakers': 1,
                                         'sweatshirt': 1,
                                         'wallet': 1}

# RELEVANT_ITEMS = {'2': 'leggings', '3': 'shorts', '4': 'blazers', '5': 'tees-and-tshirts',
#                   '8': 'womens-outerwear', '9': 'skirts', '12': 'womens-tops', '13': 'jackets', '14': 'bras',
#                   '15': 'dresses', '16': 'womens-pants', '17': 'sweaters', '18': 'womens-tops', '19': 'jeans',
#                   '20': 'leggings', '23': 'womens-top', '24': 'cardigan-sweaters', '25': 'womens-accessories',
#                   '26': 'mens-vests', '29': 'socks', '31': 'womens-intimates', '32': 'stockings',
#                   '35': 'cashmere-sweaters', '36': 'sweatshirts', '37': 'womens-suits', '43': 'mens-ties'}

# IRELEVANT_ITEMS = {'1': 'background', '6': 'bag', '7': 'shoes', '10': 'purse', '11': 'boots', '21': 'scarf',
#                    '22': 'hats', '27': 'sunglasses', '28': 'belts', '30': 'glasses', '33': 'necklace', '34': 'cape',
#                    '38': 'bracelet', '39': 'heels', '40': 'wedges', '41': 'rings',
#                    '42': 'flats', '44': 'romper', '45': 'sandals', '46': 'earrings', '47': 'gloves',
#                    '48': 'sneakers', '49': 'clogs', '50': 'watchs', '51': 'pumps', '52': 'wallets', '53': 'bodysuit',
#                    '54': 'loafers', '55': 'hair', '56': 'skin'}


filtered_paperdoll_weights_dictionary = {'womens-tops': 1,
                                         'womens-pants': 1,
                                         'shorts': 1,
                                         'jeans': 1,
                                         'jackets': 1,
                                         'blazers': 1,
                                         'skirts': 1,
                                         'dresses': 1,
                                         'sweaters': 1,
                                         'tees-and-tshirts': 1,
                                         'cardigan-sweaters': 1,
                                         'coats': 1,
                                         'womens-suits': 1,
                                         'vests': 1,
                                         'sweatshirts': 1,
                                         'v-neck-sweaters': 1,
                                         'shapewear': 1,
                                         'hosiery': 1,
                                         'leggings': 1}


def sigmoid(input_value):
    return (float(2) / (1 + np.exp(-input_value)/0.167) - 1)


def classification_rating(goldenset_classes, testset_classes, weights_dictionary):
    '''
    calculates the rating of the classes set of the test in comparison to the 'golden' (master) set of classes
    :param goldenset_classes: list of clothing classes name (strings)
    :param testset_classes: list of clothing classes name (strings)
    :return: a double, ranging from 0 to 1, rating the classification accuracy.

    steps:
    1. check that the two variables are lists of classes (words / strings / numeric, etc.). flag error if needed.
    2. check length of each. flag error if needed.
    3. detect which of 'goldenset_classes' classes exist in 'testset_classes' (positive classification - PC)
        3.1.duplicate with weighted index respectively.
        3.2. sum the product: sum(W_n*X_PC_n)
    4. count how many classes (words) of the 'goldenset_classes' are not contained in the 'testset_classes'
       (negative classification - NC)
    5. count how many classes (words) of the 'testset_classes' are not contained in the 'goldenset_classes'
       (positive wrong classification - PWC)
    6. sum the product of the 'goldenset_classes' with weighted index respectively: NWgolden = sum(W_n*X_goldenset_n)
    7. perform the rating calculation as follows, and return value:
        return_value = {(3.2) - (4) - (5)}/(6); if return_value < 0 -> return_value = 0
    '''
    # initial check (1):
    # for golden_class in goldenset_classes:
    #     if not isinstance(golden_class,str):
    #         print "goldenset_classes must be strings in all list type!"
    #         return
    #
    # for test_class in testset_classes:
    #     if not isinstance(test_class,str):
    #         print "testset_classes must be strings in all list type!"
    #         return


    # initial check (2):
    if len(goldenset_classes) == 0:
        print "goldenset_classes must not be empty!"
        return

    if len(testset_classes) == 0:
        print "testset_classes must not be empty!"
        return


    # finding matches to the goldenset (3):
    goldenset_classes = set(goldenset_classes)
    testset_classes = set(testset_classes)
    set_of_class_matches = goldenset_classes.intersection(testset_classes)


    # summing the weights of the matched classes in testset (3.1, 3.2):
    sum_weights_of_test_matches = 0
    for class_match in set_of_class_matches:
        sum_weights_of_test_matches = sum_weights_of_test_matches + weights_dictionary[class_match]
    PC = sum_weights_of_test_matches

    # how many of the goldenset are not included in the testset (4):
    NC = len(goldenset_classes) - len(set_of_class_matches)

    # how many of the testset are not included in the goldenset (5):
    PWC = len(testset_classes) - len(set_of_class_matches)

    # summing the weights of the weights for goldenset classes (6):
    sum_weights_of_goldenset_matches = 0
    for class_match in goldenset_classes:
        sum_weights_of_goldenset_matches = sum_weights_of_goldenset_matches + weights_dictionary[class_match]
    NWgolden = sum_weights_of_goldenset_matches


    # classes rating calculation (7):
    if testset_classes == 0:
        class_rating = 0.0
    elif (float(PWC) / len(testset_classes) + float(NC) / NWgolden) == 0:
        class_rating = 1.0
    else:
        class_rating = sigmoid((float(PC) / NWgolden) / (float(PWC) / len(testset_classes) + float(NC) / NWgolden))
        if class_rating < 0.0:
            class_rating = 0.0

    return class_rating


def results_rating(goldenset_images, testset_images):
    '''
    calculates the rating of the ordered images set of the test in comparison to the
    'golden' (master) set order of images.
    :param goldenset_images: list of images file names (strings)
    :param testset_images: list of images file names (strings)
    :return: a double, ranging from 0 to 1, rating the results (images set) accuracy.

    steps
    1. check that the two variables are file names (words / strings / numeric, etc.). flag error if needed.
    2. check length of each. flag error if needed.
    3. calculate: Nco = sum(if testset_images[i-y] is after testset_images[i-x] and,
                         testset_images[i-x]=goldenset_images[i-1] and,
                         testset_images[i-y]=goldenset_images[i] than,
                         x + y)
    4. calculate: Nnco = sum(if testset_images[i] is not in Nco list, and,
                          is in goldenset_images, than,
                          x = i - last_ordered_index_in_testset_image - 1
                          y = {index of (goldenset_images=testset_images[i])} -
                              {index of (goldenset_images=testset_images[i]_last_ordered) - 1}
                          x + y)
    5. calculate: Nne = sum(goldenset_image that are not in testset_images)
    6. return_value = {Ngolden - Nco - Nnco - Nne}/Ngolden; if return_value < 0 -> return_value = 0
    '''


    # initial check (1):
    # for golden_class in goldenset_images:
    #     if not isinstance(golden_class,str):
    #         print "goldenset_images must be strings in all list type!"
    #         return
    #
    # for test_class in testset_images:
    #     if not isinstance(test_class,str):
    #         print "testset_images must be strings in all list type!"
    #         return


    # # initial check (2):
    # if len(goldenset_images)==0:
    #         print "goldenset_images must not be empty!"
    #         return
    #
    # if len(testset_images)==0:
    #         print "testset_images must not be empty!"
    #         return

    images_rating = 0.0

    if goldenset_images:
        # find matching image names of golden and test, and find order of them (3):
        # we assume each image is listed only once in each list.
        index_of_ordered_at_goldenset = []
        index_of_ordered_at_testset = []
        ordered_images = []
        testset_images_tag = testset_images
        for golden_image_name in goldenset_images:
            for testset_image_name in testset_images_tag:
                if golden_image_name == testset_image_name:
                    index_of_ordered_at_goldenset.append(goldenset_images.index(golden_image_name))
                    index_of_ordered_at_testset.append(testset_images.index(golden_image_name))
                    ordered_images.append(golden_image_name)
                    testset_images_tag = testset_images[index_of_ordered_at_testset[-1]:]
                    break

        X = []
        for i in range(len(index_of_ordered_at_goldenset)):
            X.append(index_of_ordered_at_testset[i] - index_of_ordered_at_goldenset[i])
        Nco = sum(X)


        # find matching image names of golden and test, and find un-order of them (4):
        index_of_unordered_at_goldenset = []
        index_of_unordered_at_testset = []
        unordered_images = []
        last_ordered_distance = []
        weighted_unordered_distances = []
        for golden_image_name in goldenset_images:
            for testset_image_name in testset_images:
                if golden_image_name == testset_image_name and golden_image_name not in ordered_images:
                    index_of_unordered_at_goldenset.append(goldenset_images.index(golden_image_name))
                    index_of_unordered_at_testset.append(testset_images.index(golden_image_name))
                    unordered_images.append(golden_image_name)

                    for index in index_of_ordered_at_goldenset:
                        if (index - index_of_unordered_at_goldenset[-1]) > 0:
                            last_ordered_distance.append(index - 1)
                            # break
                    weighted_unordered_distances.append(last_ordered_distance[-1] - index_of_unordered_at_testset[-1])

        Y = []
        for i in range(len(index_of_unordered_at_goldenset)):
            Y.append(float((index_of_unordered_at_goldenset[i] - index_of_unordered_at_testset[i]) * \
                           weighted_unordered_distances[i]))
        Nnco = sum(Y)

        # finds how many of the goldenset are not included in the testset (5):
        Nne = len(set(goldenset_images).difference(testset_images))
        print [len(goldenset_images), Nco, Nnco, Nne]


        # classes rating calculation (6):
        images_rating = float(Nco) / len(goldenset_images) + 0.5 * float(Nnco) / len(goldenset_images) - \
                        float(Nne) / len(goldenset_images)
        if images_rating < 0.0:
            images_rating = 0.0

    return images_rating


def scorer(goldenset_classes, testset_classes, weights_dictionary, goldenset_images, testset_images):
    '''
    calculates the rating of the ordered images set of the test in comparison to the
    'golden' (master) set order of images.
    :param test_case_image_path: a path designating test image's location (string)
    :param goldenset_classes: list of classes / category's names (strings)
    :param goldenset_images: list of images file names (strings)
    :param testset_classes: list of classes / category's names which were found to exist in the image, by paperdoll (strings)
    :param testset_images: list of images file names which were found to match the test image, by paperdoll (strings)
    :param weights_dictionary: a dictionary in which the 'key's are all available classes / categories, and the values
            are float type numeric in the range of 0 to 1
    :return: a double, ranging from 0 to 1, rating the results (images set) accuracy and the category list accuracy.
    '''
    test_classes_score = classification_rating(goldenset_classes, testset_classes, weights_dictionary)
    test_results_score = results_rating(goldenset_images, testset_images)
    if test_classes_score == None:
        test_classes_score = 0.0
    if test_results_score == None:
        test_results_score = 0.0
    return test_classes_score, test_results_score


def run_scorer(test_case_image_path, goldenset_classes, goldenset_images, filtered_paperdoll=True):
    '''
    calculates the rating of the ordered images set of the test in comparison to the
    'golden' (master) set order of images.
    :param test_case_image_path: a path designating test image's location (string)
    :param goldenset_classes: list of classes / category's names (strings)
    :param goldenset_images: list of images file names (strings)
    :param weights_dictionary: a dictionary in which the 'key's are all available classes / categories, and the values
            are float type numeric in the range of 0 to 1
    :return: a double, ranging from 0 to 1, rating the results (images set) accuracy and the category list accuracy.
    '''

    if filtered_paperdoll:
        weights_dictionary = filtered_paperdoll_weights_dictionary
    else:
        weights_dictionary = original_paperdoll_weights_dictionary

    num_of_matches = 20  # of similar_results
    # resize image:
    image = test_case_image_path
    image = cv2.imread(image)
    image = background_removal.standard_resize(image, 400)[0]

    # activate paperdoll on image:
    job = paperdoll_parse_enqueue.paperdoll_enqueue(image, async=False, use_tg_worker=True)
    mask, labels, pose = job.result
    # face:
    relevance = background_removal.image_is_relevant(image)
    face = relevance.faces[0]
    testset_classes = []
    similar_results = []

    if filtered_paperdoll:  # filtered / unfiltered paperdoll
        final_mask = paperdolls.after_pd_conclusions(mask, labels, face)
        for num in np.unique(final_mask):
            # for categories score:
            category = list(labels.keys())[list(labels.values()).index(num)]
            if category in constants.paperdoll_shopstyle_women.keys():  # final mask is the PD output without the 'paperdolls.after_pd_conclusions' filtering !!!
                # task 1: get categories from image:
                testset_classes.append(category)
                item_mask = 255 * np.array(mask == num, dtype=np.uint8)
                shopstyle_cat = constants.paperdoll_shopstyle_women[category]
                # task 2: get similar results:
                if goldenset_images:  # in case of only category scoring
                    str2img = find_similar_mongo.find_top_n_results(image, item_mask, num_of_matches, shopstyle_cat)[1]
                    for element in str2img:
                        print element['_id']
                        similar_results.append(element['_id'])
    else:
        for num in np.unique(mask):
            # for categories score:
            category = list(labels.keys())[list(labels.values()).index(num)]
            if category in constants.paperdoll_shopstyle_women.keys():  # final mask is the PD output without the 'paperdolls.after_pd_conclusions' filtering !!!
                # task 1: get categories from image:
                testset_classes.append(category)
                item_mask = 255 * np.array(mask == num, dtype=np.uint8)
                # task 2: get similar results:
                if goldenset_images:  # in case of only category scoring
                    str2img = find_similar_mongo.find_top_n_results(image, item_mask, num_of_matches, category)[1]
                    for element in str2img:
                        print element['_id']
                        similar_results.append(element['_id'])

    testset_images = similar_results

    # scoring:
    test_classes_score, test_results_score = scorer(goldenset_classes, testset_classes, weights_dictionary,
                                                    goldenset_images, testset_images)

    # print weights_dictionary
    print testset_classes
    # print test_classes_score
    # print test_results_score

    return test_classes_score, test_results_score


###########
# make json file:
def fingerprint_1D(image, mask):

    roi = image
    # roi[:, :, 0] = image[:, :, 0] * (mask/255)
    # roi[:, :, 1] = image[:, :, 1] * (mask/255)
    # roi[:, :, 2] = image[:, :, 2] * (mask/255)
    # cv2.imshow('L', roi)
    # cv2.waitKey(0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
    #histograms
    devision_factor = 1
    bins = [180/devision_factor, 255/devision_factor, 255/devision_factor]
    n_pixels = cv2.countNonZero(mask)
    # weights_circumfrance = circumfrence_distance(mask)
    # numpy's histogram:
    # histHue = np.histogram(hsv[:, :, 0], bins=bins, normed=False, weights=weights_circumfrance, density=True)[0]
    # histSat = np.histogram(hsv[:, :, 1], bins=bins, normed=False, weights=weights_circumfrance, density=True)[0]
    # histInt = np.histogram(hsv[:, :, 2], bins=bins, normed=False, weights=weights_circumfrance, density=True)[0]

    # opencv's histogram:
    hist_hue = cv2.calcHist([hsv], [0], mask, [bins[0]], [0, 180])
    hist_hue = [item for sublist in hist_hue for item in sublist]  # flatten nested
    hist_hue = np.divide(hist_hue, n_pixels)

    hist_sat = cv2.calcHist([hsv], [1], mask, [bins[1]], [0, 255])
    hist_sat = [item for sublist in hist_sat for item in sublist]
    hist_sat = np.divide(hist_sat, n_pixels)

    hist_int = cv2.calcHist([hsv], [2], mask, [bins[2]], [0, 255])
    hist_int = [item for sublist in hist_int for item in sublist]  # flatten nested list
    hist_int = np.divide(hist_int, n_pixels)

    # Uniformity  t(5)=sum(p.^ 2);
    hue_uniformity = np.dot(hist_hue, hist_hue)
    sat_uniformity = np.dot(hist_sat, hist_sat)
    int_uniformity = np.dot(hist_int, hist_int)

    # Entropy   t(6)=-sum(p. *(log2(p+ eps)));
    eps = 1e-15
    max_log_value = np.log2(bins)  # this is same as sum of p log p
    l_hue = -np.log2(hist_hue + eps) / max_log_value[0]
    hue_entropy = np.dot(hist_hue, l_hue)
    l_sat = -np.log2(hist_sat + eps) / max_log_value[1]
    sat_entropy = np.dot(hist_sat, l_sat)
    l_int = -np.log2(hist_int + eps) / max_log_value[2]
    int_entropy = np.dot(hist_int, l_int)

    result_vector = [hue_uniformity, sat_uniformity, int_uniformity, hue_entropy, sat_entropy, int_entropy]
    resultVector = np.concatenate((result_vector, hist_hue, hist_sat, hist_int), axis=0)

    resultVector[6:] = resultVector[6:] * 0.05
    resultVector[6:6+bins[0]] = resultVector[6:6+bins[0]] * 0.5
    resultVector[6+bins[0]:6+bins[0]+bins[1]] = resultVector[6+bins[0]:6+bins[0]+bins[1]] * 0.225
    resultVector[6+bins[0]+bins[1]:6+bins[0]+bins[1]+bins[2]] = resultVector[6+bins[0]+bins[1]:6+bins[0]+bins[1]+bins[2]] * 0.225

    return resultVector


def do4image(path2file):
    image_dict = {'product_id': 0,
                  'photo_id':0,
                  'bbox': [],
                  '1D_fp': []}

    filename = path2file.split('/')[-1]
    if filename[-4:] == '.jpg':
        # getting data from filename string:
        file_data = filename.split('_')
        image_dict['product_id'] = file_data[1]
        image_dict['photo_id'] = file_data[3]
        bbox = [int(file_data[5]), int(file_data[6]), int(file_data[7]), int(file_data[8][:-4])]
        image_dict['bbox'] = bbox
        image_dict['1D_fp'] = []
        try:
            # adding the spatioggram data:
            image = cv2.imread(path2file)
            if not image.data:
                return image_dict
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 255
            image_dict['1D_fp'] = fingerprint_1D(image, mask).tolist()
        except:
            pass
    return image_dict


def make_json_data_at(path):
    only_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    paths = [path] * len(only_files)
    paths = [''.join(x) for x in zip(*[paths, only_files])]
    # multi-processing:
    print 'scraping data from images list @ ' + path
    # pool = multiprocessing.Pool()
    # json2list = pool.map(do4image, paths)
    json2list = []
    for file in only_files:
        json2list.append(do4image(path+file))

    output_json_filename = 'finger_print2_data_' + path.split('/')[-2] + '.json'
    print 'saving data from images list, to ' + output_json_filename + '.JSON @ run path'
    # Writing JSON data:
    j = json.dumps(json2list)
    f = open(output_json_filename, 'w')
    f.write(j)
    f.close()
    return

#loading and analysis:
def load_json_data_at(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data


def np_hist_to_cv(np_histogram_output):
    # counts, bin_edges = np_histogram_output
    counts = np_histogram_output
    return counts.ravel().astype('float32')


def chi2_distance(histA, histB):
    eps = 1e-10
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
    # return the chi-squared distance
    return d


def fingerprints_distance(query_fp, target_fp, distance_function):
    '''
    :param spaciogram_1:
    :param spaciogram_2:
    :param filter_rank:
    :return:
    '''
    ############ CHECKS ############
    # check if spaciogram_1.shape == target_fp.shape:
    rating = []
    query_fp = np_hist_to_cv(np.array(query_fp))
    target_fp = np_hist_to_cv(np.array(target_fp))
    if query_fp.shape != target_fp.shape:
        # print 'Error: the dimensions of query_fp and target_fp are not equal! \n' \
        #       'shapes are: 1st - ' + str(np.array(query_fp).shape) + '\n' \
        #       'shapes are: 2nd - ' + str(np.array(target_fp).shape)
        return rating

    query = []
    for el in query_fp:
        query.append([el])
    query_fp = query
    query = []
    for el in target_fp:
        query.append([el])
    target_fp = query

    # method = cv2.HISTCMP_BHATTACHARYYA
    # HISTCMP_CORREL Correlation
    # HISTCMP_CHISQR Chi-Square
    # HISTCMP_INTERSECT Intersection
    # HISTCMP_BHATTACHARYYA Bhattacharyya distance
    # HISTCMP_HELLINGER Synonym for HISTCMP_BHATTACHARYYA
    # HISTCMP_CHISQR_ALT
    # HISTCMP_KL_DIV
    if distance_function == 1:
        rating = cv2.compareHist(np.array(query_fp).astype('float32'), np.array(target_fp).astype('float32'), cv2.HISTCMP_CORREL)
    elif distance_function == 2:
        rating = cv2.compareHist(np.array(query_fp).astype('float32'), np.array(target_fp).astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
    # rating = chi2_distance(query_fp, target_fp)
    # rating = emd(query_fp, target_fp)
    return rating


def sort_by_fingerprint_1D(query_photo_id, data, distance_function):
    sorted_list_of_simillar_images = []
    # load the json database and find the fingequery_fprprint of the desiered image:
    query_dictionary = filter(lambda query: int(query['photo_id']) == int(query_photo_id), data)
    if not query_dictionary:
        return sorted_list_of_simillar_images
    query_fp = query_dictionary[0]['1D_fp']

    # calculate distances from query:
    for image_data in data:
        target_fp = image_data['1D_fp']
        distance = fingerprints_distance(query_fp, target_fp, distance_function)
        if distance:
            sorted_list_of_simillar_images.append([int(image_data['product_id']), int(image_data['photo_id']), distance])
            # sort the list of distances:
            sorted_list_of_simillar_images.sort(key=lambda x: x[2])

    return sorted_list_of_simillar_images


def fp_rating(product_id, sorted_list_of_simillar_images):
    product_sorted_list = np.array(sorted_list_of_simillar_images)[:, 0]
    database_size = product_sorted_list.shape[0]
    match_locations = (1 + np.where(product_sorted_list == product_id)[0]).astype('float32') / database_size
    # return nothing if there is no match:
    if len(match_locations) == 0:
        return 0

    rating = 1 - np.sum(match_locations) / match_locations.shape[0]
    return rating


##########

def lab_json():
    t = time.time()
    train_pairs_dresses_images_path = '/home/nate/Desktop/meta/dataset/train_pairs_dresses/'
    make_json_data_at(train_pairs_dresses_images_path)
    print 'elapsed time: ' + str(time.time() - t)
    t = time.time()
    train_pairs_dresses_images_path = '/home/nate/Desktop/meta/dataset/test_pairs_dresses/'
    make_json_data_at(train_pairs_dresses_images_path)
    print 'elapsed time: ' + str(time.time() - t)


def do4image_rating(image_data):
    image = image_data
    product_id = int(image['product_id'])
    sorted_list_of_simillar_images_method_1 = sort_by_fingerprint_1D(image['photo_id'], data, 1)
    sorted_list_of_simillar_images_method_2 = sort_by_fingerprint_1D(image['photo_id'], data, 2)
    rating_1 = fp_rating(product_id, sorted_list_of_simillar_images_method_1)
    rating_2 = fp_rating(product_id, sorted_list_of_simillar_images_method_2)
    return [rating_1, rating_2]

def lab_fp_rating():

    data_length = len(data)
    print 'data set length is: ' + str(data_length) + ' samples.'

    fp_methods_rating = []
    # counter = 0
    # for image in data:
    #     product_id = int(image['product_id'])
    #     sorted_list_of_simillar_images_method_1 = sort_by_fingerprint_1D(image['photo_id'], data, 1)
    #     sorted_list_of_simillar_images_method_2 = sort_by_fingerprint_1D(image['photo_id'], data, 2)
    #     rating_1 = fp_rating(product_id, sorted_list_of_simillar_images_method_1)
    #     rating_2 = fp_rating(product_id, sorted_list_of_simillar_images_method_2)
    #     fp_methods_rating.append([rating_1, rating_2])
        # counter += 1
        # if counter == 30:
        #     print 1.0 * counter / data_length

    pool = multiprocessing.Pool()
    fp_methods_rating = pool.map(do4image_rating, data)

    list_of_method_rating_listing = fp_methods_rating
    # output_json_filename = 'list_of_method_rating_listing.json'
    # Writing JSON data:
    # j = json.dumps(list_of_method_rating_listing)
    # f = open(output_json_filename, 'w')
    # f.write(j)
    # f.write(j)
    # f.close()

    np.savetxt("foo.csv", np.asarray(list_of_method_rating_listing), delimiter=",")
    print 'finished assessing distances and saved to current folder'




json_data_file_path_1 = 'finger_print2_data_test_pairs_dresses.json'
json_data_file_path_2 = 'finger_print2_data_train_pairs_dresses.json'

data1 = load_json_data_at(json_data_file_path_1)
# data2 = load_json_data_at(json_data_file_path_2)
data = data1 #+ data2






def lab_classes():
    goldenset_classes = []

    goldenset_classes.append(['skirt', 'top', 'boots'])  # 1
    goldenset_classes.append(['dress', 'belt', 'shoes', 'leggings'])  # 2
    goldenset_classes.append(['dress'])  # 3
    goldenset_classes.append(['dress', 'shoes'])  # 4
    goldenset_classes.append(['dress', 'top', 'belt', 'boots', 'bag'])  # 5
    goldenset_classes.append(['dress', 'heels'])  # 6
    goldenset_classes.append(['dress', 'belt', 'cardigan'])  # 7
    goldenset_classes.append(['dress', 'shoes'])  # 8
    goldenset_classes.append(['dress', 'shoes'])  # 9
    goldenset_classes.append(['dress', 'belt', 'cardigan', 'heels', 'bag'])  # 10
    goldenset_classes.append(['dress', 'cardigan'])  # 11
    goldenset_classes.append(['cardigan', 'belt', 'dress', 'leggings', 'shoes'])  # 12
    goldenset_classes.append(['dress', 'shoes'])  # 13
    goldenset_classes.append(['blouse', 'dress'])  # 14
    goldenset_classes.append(['dress', 'belt', 'heels'])  # 15

    ppd_classes = []
    ppd_classes.append(['belt', 'boots', 'dress', 'skirt'])  # 1
    ppd_classes.append(['bag', 'dress', 'top'])  # 2
    ppd_classes.append(['jacket', 'jeans', 'skirt'])  # 3
    ppd_classes.append(['bag', 'belt', 'dress', 'pumps', 'shoes'])  # 4
    ppd_classes.append(['bag', 'belt', 'dress', 'heels', 'shoes', 'shorts', 'skirt', 'top'])  # 5
    ppd_classes.append(['bag', 'dress', 'shirt', 'shoes'])  # 6
    ppd_classes.append(['bag', 'dress', 'jeans', 'pants', 'shoes', 't-shirt'])  # 7
    ppd_classes.append(['bag', 'dress', 'shoes'])  # 8
    ppd_classes.append(['dress', 'shoes'])  # 8
    ppd_classes.append(['bag', 'belt', 'blouse', 'dress', 'shoes', 'shorts', 'skirt', 'top'])  # 10
    ppd_classes.append(['bag', 'belt', 'cardigan', 'dress', 'shoes'])  # 11
    ppd_classes.append(['dress', 'flats', 'shoes', 'sunglasses', 'wedges'])  # 12
    ppd_classes.append(['bag', 'belt', 'dress', 'shoes'])  # 13
    ppd_classes.append(['blouse', 'dress'])  # 14
    ppd_classes.append(['belt', 'dress', 'shoes'])  # 15

    index_image_class_rating = []
    for i in range(len(goldenset_classes)):
        index_image_class_rating = classification_rating(goldenset_classes[i], ppd_classes[i], original_paperdoll_weights_dictionary)
        print index_image_class_rating





# lab_json()
lab_fp_rating()