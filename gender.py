__author__ = 'jeremy'

import logging
import copy

import numpy as np
import cv2

import background_removal
import fisherface
import Utils
import constants
from rq import Queue
from .constants import redis_conn
import time

logging.basicConfig(level=logging.DEBUG)

q = Queue('gender', connection=redis_conn)


def gender(url_or_path_or_array, threshold=0, async=False):
    job = q.enqueue('gender.genderfunc', url_or_path_or_array=url_or_path_or_array,threshold=threshold)
    start_time = time.time()
    elapsed_time = 0
    if not async:
        while job.result is None and elapsed_time<constants.gender_ttl:
            time.sleep(0.1)
            elapsed_time = time.time()-start_time
    return job.result



def genderfunc(url_or_path_or_array, threshold=0 ):
    f = fisherface.FaceRecognizer()
    f.load("faces20155727.2157.xml")  # 2500 faces

    img_arr = Utils.get_cv2_img_array(url_or_path_or_array)
    if img_arr is None:
        logging.warning('no img found in gender.py')
        return {'error':'no image found matching '+str(url_or_path_or_array)}

    cropped = crop_and_center_face(img_arr)
    if cropped is None:
        #TODO return readable error if no face is found
        logging.warning('no img returned from crop_and_center')
        cropped = img_arr

    cropped = cv2.cvtColor(cropped, constants.BGR2GRAYCONST)
#    cv2.imshow('cropped',cropped)
#    cv2.waitKey(0)
    if not (cropped.shape[0] == f.imageSize[0] and cropped.shape[1] == f.imageSize[1]):
        logging.debug('resizing ' + str(cropped.shape[:2]) + ' to expected imagesize' + str(f.imageSize))
        w = f.imageSize[0]
        h = f.imageSize[1]
        cropped = cv2.resize(cropped, (w, h))
    logging.debug('cropped final size ' + str(cropped.shape))
    print('cropped final size ' + str(cropped.shape))
    label, confidence = f.model.predict(cropped)

    if (label == '0' or label == 0) and confidence > threshold:
        return 'woman', confidence
    elif (label == '1' or label == 1) and confidence > threshold:
        return 'man', confidence
    else:
        return 'unknown', confidence


def crop_and_center_face(img_arr, face_cascade=None, eye_cascade=None, x_size=250, y_size=250):
    if not face_cascade:
        face_cascade = cv2.CascadeClassifier(constants.classifiers_folder + '/haarcascade_frontalface_default.xml')
    if not eye_cascade:
        eye_cascade = cv2.CascadeClassifier(constants.classifiers_folder + '/haarcascade_eye.xml')
    if not face_cascade:
        logging.warning('cant get face cascade in cropface in gender.py')
    if not eye_cascade:
        logging.warning('cant get eye cascade in cropface in gender.py')

    faces = background_removal.find_face(img_arr, max_num_of_faces=1)
    gray = cv2.cvtColor(img_arr, constants.BGR2GRAYCONST)
    i = 0
    cropped = None
    if faces is not None:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 2:
                left = eyes[0]
                right = eyes[1]
                if left[0] > right[0]:
                    temp = left
                    left = right
                    right = temp
                original_left = [left[0] + x, left[1] + y, left[2], left[3]]
                original_right = [right[0] + x, right[1] + y, right[2], right[3]]
                lined_up = line_up_eyes(img_arr, original_left, original_right)
                lined_up_roi = lined_up[0:y_size, 0:x_size]
                dst = copy.copy(lined_up_roi)  # rotated.copy.deepcopy()
                return dst
            if i == 0:
                cropped = img_arr[y:y + h, x:x + w]
            i += 1

        # no set of two eyes found so just return as is
        dst = copy.copy(cropped)  # rotated.copy.deepcopy()
        return dst

    else:
        return img_arr


def line_up_eyes(img_arr, left_eye_box, right_eye_box, expected_left_eye_center=[100, 115],
                 expected_right_eye_center=[150, 115]):
    center_left = [left_eye_box[0] + int(float(left_eye_box[2]) / 2.0),
                   left_eye_box[1] + int(float(left_eye_box[3]) / 2.0)]
    center_right = [right_eye_box[0] + int(float(right_eye_box[2]) / 2.0),
                    right_eye_box[1] + int(float(right_eye_box[3]) / 2.0)]

    # translation of left eye
    logging.debug('l.eye:' + str(left_eye_box) + ' center:' + str(center_left))
    logging.debug('r.eye:' + str(right_eye_box) + ' center:' + str(center_right))

    Dx = expected_left_eye_center[0] - center_left[0]
    Dy = expected_left_eye_center[1] - center_left[1]
    M = np.array([[1, 0, Dx], [0, 1, Dy]], dtype=np.float32)
    rows, cols = img_arr.shape[:2]
    logging.debug('M:' + str(M) + ' colsXrows:' + str(cols) + ',' + str(rows))
    xlated = cv2.warpAffine(img_arr, M, (cols, rows))

    dx = center_right[0] - center_left[0]
    dy = center_right[1] - center_left[1]
    logging.debug('dx,dy:' + str(dx) + ',' + str(dy))
    theta = np.arctan((float(dy) / dx))
    theta_degrees = theta * 180.0 / 3.1415
    scale = float(expected_right_eye_center[0] - expected_left_eye_center[0]) / (center_right[0] - center_left[0])
    angle = theta_degrees
    center = (expected_left_eye_center[0], expected_left_eye_center[1])
    M = cv2.getRotationMatrix2D(center, angle, scale)
    logging.debug('center ' + str(center))
    logging.debug('scale ' + str(scale))
    logging.debug('theta:' + str(theta) + ' deg:' + str(theta_degrees) + ' M:' + str(M))

    rotated = cv2.warpAffine(xlated, M, (cols, rows))

    return rotated


if __name__ == "__main__":

    img_name = 'images/female1.jpg'
    g = gender(img_name)
    print('trying gender queue:' + img_name+' seems to be '+ str(g))
    img_name = 'images/male1.jpg'
    g = gender(img_name)
    print('trying gender queue:' + img_name+' seems to be '+ str(g))
    img_name = 'images/male2.jpg'
    g = gender(img_name)
    print('trying gender queue:' + img_name+' seems to be '+ str(g))
    img_name = 'images/male3.jpg'
    g = gender(img_name)
    print('trying gender queue:' + img_name+' seems to be '+ str(g))


#    img_name = 'images/female1.jpg'
#    img_arr = cv2.imread(img_name)
#    g = genderfunc(img_name)
#    print('trying genderfunc:' + img_name+' seems to be '+ str(g))
#    cv2.imshow('original',img_arr)
#    cv2.waitKey(0)

    '''
some output from lfw db to get a sense of where the eyes are

in LFW , eye centers at:
100 115    143 113
98 116    150 116
105 115   146 114
102 114   147 115
102 113  144 112
101 114   148 113
[100 115  150 115]

facedetections from haarcascade:
1leftmean[ 20.71270718  30.99309392  28.78453039  28.78453039]
tmean[ 62.77348066  30.18922652  28.28314917  28.28314917]


left:[21 34 29 29] right:[64 33 28 28]
left:[24 29 33 33] right:[71 31 25 25]
left:[19 28 26 26] right:[59 28 26 26]
left:[25 32 31 31] right:[66 32 28 28]
left:[26 39 26 26] right:[39 75 52 52]
left:[16 25 32 32] right:[57 24 32 32]
left:[30 31 27 27] right:[63 29 26 26]
left:[16 26 28 28] right:[55 17 32 32]
left:[20 30 28 28] right:[69 27 32 32]
left:[13 23 28 28] right:[67 22 30 30]
left:[19 30 29 29] right:[60 30 29 29]
left:[16 27 28 28] right:[61 25 32 32]
left:[19 35 23 23] right:[64 31 26 26]
left:[20 26 30 30] right:[60 27 25 25]
left:[23 32 29 29] right:[64 31 27 27]
left:[22 27 34 34] right:[67 33 24 24]
left:[26 30 31 31] right:[65 32 25 25]
left:[33 72 42 42] right:[61 32 30 30]
left:[24 32 30 30] right:[75 33 24 24]
left:[20 30 27 27] right:[59 31 25 25]
left:[22 31 32 32] right:[66 32 29 29]
left:[20 30 29 29] right:[61 32 26 26]
left:[23 32 25 25] right:[64 30 29 29]
left:[16 20 34 34] right:[52 17 38 38]
left:[17 27 24 24] right:[58 29 23 23]
left:[20 28 27 27] right:[27 29 27 27]
left:[20 25 35 35] right:[32 33 22 22]
left:[26 32 36 36] right:[76 38 26 26]
left:[18 31 21 21] right:[60 26 26 26]

code to generate it

    BASE_PATH = os.getcwd()
    BASE_PATH = os.path.join(BASE_PATH, 'female')
    print('basepath:' + BASE_PATH)
    lefts=[]
    rights=[]
    i=0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                img_arr = cv2.imread(abs_path)
                #cv2.imshow('or',img_arr)
                #cv2.waitKey(0)
                c,e = cropFace(img_arr)
                if c is not None:
                    if len(e)==2:
                        first = e[0]
                        second = e[1]
                        if first[0]>second[0]:
                            temp = first
                            first = second
                            second = temp
                        print('left:'+str(first)+' right:'+str(second))
                        lefts.append(first)
                        rights.append(second)
                        i = i + 1
                        print('leftmean'+str(np.mean(lefts,0)))
                        print('rtmean'+str(np.mean(rights,0)))
                #MEN
    f2.close()
    f2 = open('men' + csv_filename, 'w')
    BASE_PATH = os.getcwd()
    BASE_PATH = os.path.join(BASE_PATH, 'male')
    print('basepath:' + BASE_PATH)
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                s = "{0}{1}{2}\n".format(abs_path, SEPARATOR, label)
                print(str(s))


    '''