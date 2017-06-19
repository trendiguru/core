__author__ = 'Nadav Paz'
# Libraries import
# TODO - combine pose-estimation face detection as a backup to the cascades face detection

import string
import collections
import os
import dlib
import cv2
import numpy as np
import rq
from . import constants
from . import Utils
from . import ccv_facedetector as ccv
from . import kassper
import time
from functools import partial
import sklearn
from matplotlib import pyplot as plt

detector = dlib.get_frontal_face_detector()
db = constants.db
redis_conn = constants.redis_conn


def image_is_relevant(image, use_caffe=False, image_url=None):
    """
    main engine function of 'doorman'
    :param image: nXmX3 dim ndarray representing the standard resized image in BGR colormap
    :return: namedtuple 'Relevance': has 2 fields:
                                                    1. isRelevant ('True'/'False')
                                                    2. faces list sorted by relevance (empty list if not relevant)
    Thus - the right use of this function is for example:
    - "if image_is_relevant(image).is_relevant:"
    - "for face in image_is_relevant(image).faces:"
    """
    Relevance = collections.namedtuple('relevance', 'is_relevant faces')
    faces_dict = find_face_dlib(image, 3)
    # faces_dict = find_face_cascade(image, 10)
    # if len(faces_dict['faces']) == 0:
    #     faces_dict = find_face_ccv(image, 10)
    if not faces_dict['are_faces']:
        # if use_caffe:
        # return Relevance(caffeDocker_test.is_person_in_img('url', image_url).is_person, [])
        # else:
        return Relevance(False, [])
    else:
        return Relevance(True, faces_dict['faces'])


def find_face_ccv(image_arr, max_num_of_faces=100):
    if not isinstance(image_arr, np.ndarray):
        raise IOError('find_face got a bad input: not np.ndarray')
    else:  # do ccv
        faces = ccv.ccv_facedetect(image_array=image_arr)
        if faces is None or len(faces) == 0:
            return {'are_faces': False, 'faces': []}
        else:
            return {'are_faces': True, 'faces': choose_faces(image_arr, faces, max_num_of_faces)}


def find_face_cascade(image, max_num_of_faces=10):
    gray = cv2.cvtColor(image, constants.BGR2GRAYCONST)
    face_cascades = [
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt2.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_alt_tree.xml')),
        cv2.CascadeClassifier(os.path.join(constants.classifiers_folder, 'haarcascade_frontalface_default.xml'))]
    cascade_ok = False
    for cascade in face_cascades:
        if not cascade.empty():
            cascade_ok = True
            break
    if cascade_ok is False:
        raise IOError("no good cascade found!")
    faces = []
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(5, 5),
            flags=constants.scale_flag
        )
        if len(faces) > 0:
            break
    if len(faces) == 0:
        return {'are_faces': False, 'faces': []}
    return {'are_faces': True, 'faces': choose_faces(image, faces, max_num_of_faces)}


def find_face_dlib(image, max_num_of_faces=10):
    faces = detector(image, 1)
    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    print('dlib found {} faces'+str(len(faces)))
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    final_faces = choose_faces(image, faces, max_num_of_faces)
    return {'are_faces': len(final_faces) > 0, 'faces': final_faces}

def find_face_dlib_with_scores(image, max_num_of_faces=100):
    '''
    return the full info including scores
    :param image:
    :param max_num_of_faces:
    :return:
    '''
    start = time.time()
    if isinstance(image,basestring):
        image = Utils.get_cv2_img_array(image)
   ## faces, scores, idx = detector.run(image, 1, -1) - gives more results, those that add low confidence percentage ##
    ## faces, scores, idx = detector.run(image, 1, 1) - gives less results, doesn't show the lowest confidence percentage results ##
    ## i can get only the faces locations with: faces = detector(image, 1) ##
    faces, scores, idx = detector.run(image, 1)

    for i, d in enumerate(faces):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))

    print("dlib found {} faces in {} s.".format(len(faces),(time.time() - start)))

    faces = [[rect.left(), rect.top(), rect.width(), rect.height()] for rect in list(faces)]
    if not len(faces):
        return {'are_faces': False, 'faces': []}
    #final_faces = choose_faces(image, faces, max_num_of_faces)
    print "number of faces: {0}\n".format(len(faces))
    return {'are_faces': len(faces) > 0, 'faces': faces, 'scores': scores}


def choose_faces(image, faces_list, max_num_of_faces):
    # in faces w = h, so biggest face will have the biggest h (we could also take w)
    biggest_face = 0
    if not isinstance(faces_list, list):
        faces_list = faces_list.tolist()

    faces_list.sort(key=lambda x: x[3], reverse=True)  # sort the faces from big to small according to the height (which is also the width)

    relevant_faces = []
    for face in faces_list:
        if face_is_relevant(image, face):
            # since the list is reversed sorted, the first relevant face, will be the biggest
            if biggest_face == 0:
                biggest_face = face[3]
            # in case the current face is not the biggest relevant one, i'm going to check if its height (= wight) * 1.6 smaller
            # than the biggest face's height, if so, the current face is not relevant and also the next
            # (which are smaller)
            else:
                if 1.6 * face[3] < biggest_face:
                    break

            relevant_faces.append(face)

    # relevant_faces = [face for face in faces_list if face_is_relevant(image, face)]

    if len(relevant_faces) > max_num_of_faces:
        score_face_local = partial(score_face, image=image)
        relevant_faces.sort(key=score_face_local)
        relevant_faces = relevant_faces[:max_num_of_faces]
    return relevant_faces


def score_face(face, image):
    image_height, image_width, d = image.shape
    optimal_face_point = int(image_width / 2), int(0.125 * image_height)
    optimal_face_width = 0.1 * max(image_height, image_width)
    x, y, w, h = face
    face_centerpoint = x + w / 2, y + h / 2
    # This is the distance from face centerpoint to optimal centerpoint.
    positon_score = np.linalg.norm(np.array(face_centerpoint) - np.array(optimal_face_point))
    size_score = abs((float(w) - optimal_face_width))
    total_score = 0.6 * positon_score + 0.4 * size_score
    return total_score


def face_is_relevant(image, face):
    # (x,y) - left upper coordinates of the face, h - height of face, w - width of face
    # face relevant if:
    # - face bounding box is all inside the image
    # - h > 7% from the full image height and h < 25% from the full image height
    # - all face (height wise) is above the middle of the image
    # - if we see enough from the body - at least 4.7 "faces" (long) beneath the end of the face (y + h) - we'will need to delete this condition when we'll know to handle top part of body by its own
    # - face inside border of 6% from each side of the right and left of the full image
    # - face have to be with blurry > 100
    # - skin pixels (according to our constants values) are more than third of all the face pixels
    image_height, image_width, d = image.shape
    x, y, w, h = face
    face_image = image[y:y + h, x:x + w, :]
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    print "face: {0}, blurry: {1}".format(face, variance_of_laplacian(gray_face))
    # threshold = face + 4.7 faces down = 5.7 faces
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    face_ycrcb = ycrcb[y:y + h, x:x + w, :]
    if (x > 0 and x + w < image_width and y > 0 and y + h < image_height) \
            and 0.07 * image.shape[0] < h < 0.25 * image.shape[0] \
            and y < (image.shape[0] / 2) - h \
            and (image.shape[0] - (h * 4.7)) > (y + h) \
            and (0.06 * image.shape[1] < x and 0.94 * image.shape[1] > (x + w)) \
            and variance_of_laplacian(gray_face) > 225 \
            and is_skin_color(face_ycrcb):
        return True
    else:
        return False


def is_skin_color(face_ycrcb):
    # return True if skin pixels (according to our constants values) are more
    # than third of all the face pixels
    h, w, d = face_ycrcb.shape
    if not w*h:
        return False
    num_of_skin_pixels = 0
    for i in range(0, h):
        for j in range(0, w):
            cond = face_ycrcb[i][j][0] > 0 and 131 < face_ycrcb[i][j][1] < 180 and 80 < face_ycrcb[i][j][2] < 130
            if cond:
                num_of_skin_pixels += 1
    return num_of_skin_pixels / float(h * w) > 0.33


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    try:
        blurry = cv2.Laplacian(image, cv2.CV_64F).var()
    except AttributeError:
        return False

    return blurry


def is_one_color_image(image):
    # convert RGB to HSV then calculate the standard deviation of the hsv_image,
    # and only refer to std of H (Hue - color)
    std_threshold = 20

    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean, std = cv2.meanStdDev(hsv_image)
    except AttributeError:
        # in case of an AttributeError i return False which mean proceed as if the image
        # is good to go, i did it cause i guess it's not the right place for that kind of
        # filter here
        return False

    if std[0][0] < std_threshold:
        return True
    else:
        return False


def average_bbs(bb1, bb2):
    bb_x = int((bb1[0] + bb2[0]) / 2)
    bb_y = int((bb1[1] + bb2[1]) / 2)
    bb_w = int((bb1[2] + bb2[2]) / 2)  # this isn't necessarily width, it could be x2 if rect is [x1,y1,x2,y2]
    bb_h = int((bb1[3] + bb2[3]) / 2)

    bb_out = [bb_x, bb_y, bb_w, bb_h]
    return bb_out
    # bb_out = int(np.divide(bb1[:]+bb2[:],2))


def combine_overlapping_rectangles(bb_list):
    if len(bb_list) < 2:
        return bb_list
    iou_threshold = 0.8  # TOTALLY ARBITRARY THRESHOLD
    for i in range(0, len(bb_list)):
        for j in range(i + 1, len(bb_list)):
            bb1 = bb_list[i]
            bb2 = bb_list[j]
            iou = Utils.intersectionOverUnion(bb1, bb2)
            if iou > iou_threshold:
                print('combining bbs')
                bb_new = average_bbs(bb1, bb2)
                # bb_list.remove(bb1)
                print('bblist before ' + str(bb_list))
                bb_list = np.delete(bb_list, j, axis=0)
                bb_list = np.delete(bb_list, i, axis=0)
                bb_list = np.append(bb_list, bb_new, axis=0)
                print('bblist after ' + str(bb_list))

                return (combine_overlapping_rectangles(bb_list))
            else:
                print('iou too small, taking first bb')
                print('bblist before ' + str(bb_list))
                bb_list = np.delete(bb_list, j, axis=0)
                print('bblist after ' + str(bb_list))
                return (combine_overlapping_rectangles(bb_list))

    return (bb_list)


def body_estimation(image, face):
    x, y, w, h = face
    y_down = image.shape[0] - 1
    x_back = np.max([x - 2 * w, 0])
    x_back_near = np.max([x - w, 0])
    x_ahead = np.min([x + 3 * w, image.shape[1] - 1])
    x_ahead_near = np.min([x + 2 * w, image.shape[1] - 1])
    rectangles = {"BG": [], "FG": [], "PFG": [], "PBG": []}
    rectangles["FG"].append([x, x + w, y, y + h])  # face
    rectangles["PFG"].append([x, x + w, y + h, y_down])  # body
    rectangles["BG"].append([x, x + w, 0, y])  # above face
    rectangles["BG"].append([x_back, x, 0, y + h])  # head left
    rectangles["BG"].append([x + w, x_ahead, 0, y + h])  # head right
    rectangles["PFG"].append([x_back_near, x, y + h, y_down])  # left near
    rectangles["PFG"].append([x + w, x_ahead_near, y + h, y_down])  # right near
    if x_back_near > 0:
        rectangles["PBG"].append([x_back, x_back_near, y + h, y_down])  # left far
    if x_ahead_near < image.shape[1] - 1:
        rectangles["PBG"].append([x_ahead_near, x_ahead, y + h, y_down])  # right far
    return rectangles


def bb_mask(image, bounding_box):
    if isinstance(bounding_box, basestring):
        bb_array = [int(bb) for bb in string.split(bounding_box)]
    else:
        bb_array = bounding_box
    image_w = image.shape[1]
    image_h = image.shape[0]
    x, y, w, h = bb_array
    y_down = np.min([image_h-1, y+1.2*h])
    x_back = np.max([x-0.2*w, 0])
    y_up = np.max([0, y-0.2*h])
    x_ahead = np.min([image_w-1, x+1.2*w])
    rectangles = {"BG": [], "FG": [], "PFG": [], "PBG": []}
    rectangles["PFG"].append([int(x), int(x+w), int(y), int(y+h)])
    rectangles["PBG"].append([int(x_back), int(x_ahead), int(y_up), int(y_down)])
 #   print('bb_mask rectangles {} imw {} imh {} x {} y {} w {} h {} yd {} yu {} xd {} xu {} '.format(rectangles,image_w,image_h,x,y,w,h,y_down,y_up,x_back,x_ahead))
    mask = create_mask_for_gc(rectangles, image)
    return mask


def paperdoll_item_mask(item_mask, bb):
    x, y, w, h = bb
    mask_h, mask_w = item_mask.shape
    mask = np.zeros(item_mask.shape, dtype=np.uint8)
    y_down = np.min([mask_h - 1, y + 1.1 * h])
    x_back = np.max([x - 0.1 * w, 0])
    y_up = np.max([0, y - 0.1 * h])
    x_ahead = np.min([mask_w - 1, x + 1.1 * w])
    mask[y_up:y_down, x_back:x_ahead] = 3
    mask = np.where(item_mask != 0, 1, mask)
    return mask


def create_mask_for_gc(rectangles, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for rectangle in rectangles["BG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 0
        print('BG'+str(rectangle))
    for rectangle in rectangles["PBG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 2
        print('PBG'+str(rectangle))
    for rectangle in rectangles["PFG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 3
        print('PFG'+str(rectangle))
    for rectangle in rectangles["FG"]:
        x0, x1, y0, y1 = rectangle
        mask[y0:y1, x0:x1] = 1
        print('FG'+str(rectangle))
    return mask


def create_arbitrary(image):
    h, w = image.shape[:2]
    mask = np.zeros([h, w], dtype=np.uint8)
    sub_h = h / 20
    sub_w = w / 10
    mask[2 * sub_h:18 * sub_h, 2 * sub_w:8 * sub_w] = 2
    mask[4 * sub_h:16 * sub_h, 3 * sub_w:7 * sub_w] = 3
    mask[7 * sub_h:13 * sub_h, 4 * sub_w:6 * sub_w] = 1
    return mask


def standard_resize(image, max_side):
    original_w = image.shape[1]
    original_h = image.shape[0]
    if image.shape[0] < max_side and image.shape[1] < max_side:
        return image, 1
    aspect_ratio = float(np.amax((original_w, original_h))/float(np.amin((original_h, original_w))))
    resize_ratio = float(float(np.amax((original_w, original_h))) / max_side)
    if original_w >= original_h:
        new_w = max_side
        new_h = max_side/aspect_ratio
    else:
        new_h = max_side
        new_w = max_side/aspect_ratio
    resized_image = cv2.resize(image, (int(new_w), int(new_h)))
    return resized_image, resize_ratio


def resize_back(image, resize_ratio):
    w = image.shape[1]
    h = image.shape[0]
    new_w = w*resize_ratio
    new_h = h*resize_ratio
    resized_image = cv2.resize(image, (int(new_w), int(new_h)))
    return resized_image


def get_fg_mask(image, bounding_box=None):
    rect = (0, 0, image.shape[1]-1, image.shape[0]-1)
    bgdmodel = np.zeros((1, 65), np.float64)  # what is this wierd size about? (jr)
    fgdmodel = np.zeros((1, 65), np.float64)

    # bounding box was sent from a human - grabcut with bounding box mask
    if Utils.legal_bounding_box(bounding_box):
        if Utils.all_inclusive_bounding_box(image, bounding_box):  # bb is nearly the whole image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
        else:
            mask = bb_mask(image, bounding_box)
            cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    # grabcut on the whole image, with/without face
    else:
        faces_dict = find_face_cascade(image)
        # if len(faces_dict['faces']) > 0:  # grabcut with mask
        #     try:
        #         rectangles = body_estimation(image, faces_dict['faces'][0])
        #         mask = create_mask_for_gc(rectangles, image)
        #     except:
        #         mask = create_mask_for_gc(image)
        #
        # else:  # grabcut with arbitrary rect
        mask = create_arbitrary(image)
        cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    return mask2


def get_masked_image(image, mask):
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def image_white_bckgnd(image, mask):
    for i in range(0, np.shape(image)[0]):
        for j in range(0, np.shape(image)[1]):
            if mask[i][j] == 0:
                image[i][j][0] = 255
                image[i][j][1] = 255
                image[i][j][2] = 255
    return image


def get_binary_bb_mask(image, bb=None):
    """
    The function returns a ones mask within the bb regions, and an image-size ones matrix in case of None bb
    :param image:
    :param bb:
    :return:
    """
    if (bb is None) or (bb == np.array([0, 0, 0, 0])).all():
        return np.ones((image.shape[1], image.shape[0]))
    x, y, w, h = bb
    bb_masked = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    bb_masked[y:y+h, x:x+w] = 255
    return bb_masked


def face_skin_color_estimation(image, face_rect):
    x, y, w, h = face_rect
    face_image = image[y:y + h, x:x + w, :]
    face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    bins = 180
    n_pixels = face_image.shape[0] * face_image.shape[1]
    hist_hue = cv2.calcHist([face_hsv], [0], None, [bins], [0, 180])
    hist_hue = np.divide(hist_hue, n_pixels)
    skin_hue_list = []
    for l in range(0, 180):
        if hist_hue[l] > 0.013:
            skin_hue_list.append(l)
    return skin_hue_list


def face_skin_color_estimation_gmm(image, face_rect,visual_output=False):
    '''
    get params of skin color - gaussian approx for h,s,v (independently)
    :param image:
    :param face_rect:
    :param visual_output:
    :return: [(h_mean,h_std),(s_mean,s_std),(v_mean,v_std)]
    '''

    x, y, w, h = face_rect
    face_image = image[y:y + h, x:x + w, :]
  # face_hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    face_YCrCb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
    n_pixels = face_image.shape[0]*face_image.shape[1]
    print('npixels:'+str(n_pixels))
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    # Define some test data which is close to Gaussian
    gmm = sklearn.mixture.GMM()
#    r = gmm.fit(face_hsv) # GMM requires 2D data as of sklearn version 0.16
    channels = [np.ravel(face_YCrCb[:,:,0]),np.ravel(face_YCrCb[:,:,1]),np.ravel(face_YCrCb[:,:,2])]
    labels = ['Y','Cr','Cb']
    results = []
    for data,label in zip(channels,labels):
        r = gmm.fit(data[:,np.newaxis]) # GMM requires 2D data as of sklearn version 0.16
        print("mean : %f, var : %f" % (r.means_[0, 0], r.covars_[0, 0]))
        results.append((r.means_[0, 0], np.sqrt(r.covars_[0, 0])))
#        p0 = [1., 0., 1.]
#        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        # Get the fitted curve

        if visual_output:
            hist, bin_edges = np.histogram(data, density=False)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
            plt.plot(bin_centres, hist,'.-', label='Test data '+label)
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
    if visual_output:
        plt.legend()
        plt.show()
    return results


def simple_mask_grabcut(image, rect=None, mask=None):
    if rect is None:
        rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
        mode = cv2.GC_INIT_WITH_MASK
    else:
        rect = tuple(rect)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mode = cv2.GC_INIT_WITH_RECT
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, mode)
    outmask = np.where((mask == 1) + (mask == 3), 255, 0)
    return outmask


def person_isolation(image, face):
    x, y, w, h = face
    image_copy = np.zeros(image.shape, dtype=np.uint8)
    x_back = np.max([x - 1.5 * w, 0])
    x_ahead = np.min([x + 2.5 * w, image.shape[1] - 2])
    image_copy[:, int(x_back):int(x_ahead), :] = image[:, int(x_back):int(x_ahead), :]
    return image_copy


def create_non_face_dresses(kw):
    if kw not in db.collection_names():
        db.create_collection(kw)
    if kw == 'mini_with_face':
        curs = db.ShopStyle_Female.find({'$or':
                                         [{'$and': [{'longDescription': {'$regex': ' mini'}}, {'categories': 'dress'}]},
                                         {'$and': [{'longDescription': {'$regex': 'Mini'}}, {'categories': 'dress'}]},
                                         {'$and': [{'longDescription': {'$regex': 'Mini-'}}, {'categories': 'dress'}]},
                                         {'$and': [{'longDescription': {'$regex': 'mini-'}}, {'categories': 'dress'}]}]})
        skin_thresh = 0.05
    elif kw == 'maxi_with_face':
        curs = db.ShopStyle_Female.find({'$and': [{'longDescription': {'$regex': ' maxi '}}, {'categories': 'dress'}]})
        skin_thresh = 0.02
    cnt = 0
    inserted = 0
    print "total docs = {0}".format(curs.count())
    for doc in curs:
        cnt += 1
        image = Utils.get_cv2_img_array(doc['images']['XLarge'])
        if image is not None:
            try:
                faces = find_face_dlib(image)
                #if not faces['are_faces'] and check_skin_percentage(image) < skin_thresh:
                if faces['are_faces']:
                    db[kw].insert_one({'image_url': doc['images']['XLarge']})
                    inserted += 1
                    print "inserted {0}/{1}".format(inserted, cnt)
            except Exception as e:
                print str(e)


def check_skin_percentage(image):
    skin_mask = kassper.skin_detection_with_grabcut(image, image, skin_or_clothes='skin')
    return float(cv2.countNonZero(skin_mask))/(image.shape[0]*image.shape[1])
