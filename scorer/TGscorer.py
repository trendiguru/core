import numpy as np
import find_similar_mongo
import background_removal
import constants
import Utils
from .paperdoll import paperdoll_parse_enqueue
import paperdolls
import cv2

def classification_rating(goldenset_classes,testset_classes,weights_dictionary):
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
    if len(goldenset_classes)==0:
            print "goldenset_classes must not be empty!"
            return

    if len(testset_classes)==0:
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
    class_rating = float(PC)/NWgolden - float(NC)/NWgolden - float(PWC)/len(testset_classes)
    if class_rating < 0.0:
        class_rating = 0.0

    return class_rating

def results_rating(goldenset_images,testset_images):
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


    # initial check (2):
    if len(goldenset_images)==0:
            print "goldenset_images must not be empty!"
            return

    if len(testset_images)==0:
            print "testset_images must not be empty!"
            return


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
                        last_ordered_distance.append(index-1)
                        # break
                weighted_unordered_distances.append(last_ordered_distance[-1] - index_of_unordered_at_testset[-1])

    Y = []
    for i in range(len(index_of_unordered_at_goldenset)):
        Y.append(float((index_of_unordered_at_goldenset[i] - index_of_unordered_at_testset[i]) * \
               weighted_unordered_distances[i]))
    Nnco = sum(Y)

    # finds how many of the goldenset are not included in the testset (5):
    Nne = len(set(goldenset_images).difference(testset_images))
    print [len(goldenset_images),Nco,Nnco,Nne]


    # classes rating calculation (6):
    images_rating = float(Nco)/len(goldenset_images) + 0.5*float(Nnco)/len(goldenset_images) - \
                    float(Nne)/len(goldenset_images)
    if images_rating < 0.0:
        images_rating = 0.0

    return images_rating

def score(goldenset_classes,testset_classes,weights_dictionary,goldenset_images,testset_images):
    test_classes_score = classification_rating(goldenset_classes,testset_classes,weights_dictionary)
    test_results_score = results_rating(goldenset_images,testset_images)
    return test_classes_score, test_results_score

def run_scorer(test_case_image_path,goldenset_classes,goldenset_images,weights_dictionary):



    num_of_matches = 20 # of similar_results
    # resize image:
    image = test_case_image_path
    image = cv2.imread(image)
    image = background_removal.standard_resize(image, 400)[0]

    # activate paperdoll on image:
    mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(image, False)

    # task 1: get categories from image:
    # face:
    relevance = background_removal.image_is_relevant(image)
    face = relevance.faces[0]
    final_mask = paperdolls.after_pd_conclusions(mask, labels, face)
    testset_classes = []
    for num in np.unique(final_mask):
        # for categories score:
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category in constants.paperdoll_shopstyle_women.keys():
            testset_classes.append(category)

    # task 2: get similar results:
        item_mask = 255 * np.array(mask == num, dtype=np.uint8)
        shopstyle_cat = constants.paperdoll_shopstyle_women[category]
        similar_results = find_similar_mongo.find_top_n_results(image,item_mask,num_of_matches,shopstyle_cat)[1]



    print weights_dictionary
    print testset_classes
    print classification_rating(goldenset_classes,testset_classes,weights_dictionary)



        # scoring:
        # test_classes_score, test_results_score = scorer(goldenset_classes,testset_classes,weights_dictionary,goldenset_images,testset_images)
