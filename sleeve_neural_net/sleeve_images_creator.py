import cv2
import numpy as np
from PIL import Image, ImageEnhance
# import scipy as sp
import os
# import argparse
# from __future__ import print_function
#
# def adjust_gamma(image, gamma=1.0):
# 	# build a lookup table mapping the pixel values [0, 255] to
# 	# their adjusted gamma values
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
#
# 	# apply gamma correction using the lookup table
# 	return cv2.LUT(image, table)
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# args = vars(ap.parse_args())

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

def whiten_image(image):
    '''
    :param image: uint8 image grayscale or BGR
    :return:
    '''
    whitend_image = []
    # if grayscale:
    if len(image.shape) == 1:
        oMax = image.max()
        nMax = 255
        dScale = nMax - oMax
        if dScale > 0:
            whitend_image = np.zeros(image.shape)
            oMin = image.min()
            nMin = oMin
            whitend_image = remap(image, oMin, oMax, nMin, nMax)
        else:
            whitend_image = image

    # if BGR:
    elif len(image.shape) == 3:
        oMax = image.max()
        nMax = 255
        dScale = nMax - oMax
        if dScale > 0:
            whitend_image = np.zeros(image.shape)
            for i in range(3):
                oMin = image[:, :, i].min()
                nMin = oMin
                oMax = image[:, :, i].max()
                nMax = oMax + dScale
                whitend_image[:, :, i] = remap(image[:, :, i], oMin, oMax, nMin, nMax)
        else:
            whitend_image = image

    else:
        print 'Error: input is not a 3 channle image nore a grayscale (1 ch) image!'
    return whitend_image.astype('uint8')

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)



# face_cascade = cv2.CascadeClassifier('/home/netanel/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('/home/developer/python-packages/trendi/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('/home/nate/Desktop/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# current_directory = os.path.abspath()

enhancement_factors = [0.5, 1.0, 1.1]
image_file_types = ['.jpg', 'jpeg', '.png', '.bmp', '.gif']
a = 1.35 # scalar for increasisleeve box in relation to face box (1==100%)
max_angle = 10 # tilt angle of the image for diversification
angle_offset = 9 # tilt angle of the image for diversification
# max_offset = 0.01 # maximum horizontal movement (% (out of box X) of tsleeve box for diversification
# delta_offset = max_offset # horizontal movement increments(%)
output_images_size = (32, 32) # pixels^2
dataset_directory_name = 'dataset'


current_directory_name = os.getcwd()
directory_path = current_directory_name + '/' + dataset_directory_name
if not os.path.exists(directory_path):
    os.mkdir(dataset_directory_name)

my_path = os.path.dirname(os.path.abspath(__file__)) + '/'
neck_type_images_directory = ['no_sleeve', 'short_sleeve', 'long_sleeve']
# offset_range = np.arange(-max_offset, max_offset * 1.01, delta_offset)
# print offset_range
max_number_of_samples_of_smallest_type = 0
for type in neck_type_images_directory:
    my_path_type = my_path + type + '/'
    only_files = [f for f in os.listdir(my_path_type) if os.path.isfile(os.path.join(my_path_type, f))]
    if max_number_of_samples_of_smallest_type > len(only_files) or max_number_of_samples_of_smallest_type == 0:
        max_number_of_samples_of_smallest_type = len(only_files)

print max_number_of_samples_of_smallest_type
only_files = []
a = (a-1)/2
for type in neck_type_images_directory:
    my_path_type = my_path + type + '/'
    print my_path_type
    only_files = [f for f in os.listdir(my_path_type) if os.path.isfile(os.path.join(my_path_type, f))]
    only_image_files = []
    for file_name in only_files[:max_number_of_samples_of_smallest_type]:
        for image_type in image_file_types:
            if image_type in file_name:
                only_image_files.append(my_path_type + file_name)

    image_number = 0
    for image_file_name in only_image_files:
        print image_file_name
        image = cv2.imread(image_file_name, 1)
        face = face_cascade.detectMultiScale(image, 1.1, 2)
        # checking if the face (ancore) is present / detected:
        if len(face) == 0:
            continue
        face = face[0]
        offsetted_face = face
        row, col, dep = image.shape

        # no flip along vertical axis:sleeve_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
        flippsleeve_image_center_point = (col - (face[0]+0.5*face[2]), face[1]+8*face[3])
        # for offset1 in offset_range:
        #     offsetted_face[0] = face[0] #+ offset1 * face[2]
            # for offset2 in offset_range:
            # offsetted_face[1] = face[1] #+ offset2 * face[3]
        for angle in range(-max_angle, max_angle+1, angle_offset):
            rotated_image_matrix = cv2.getRotationMatrixsleeve_image_center_point, angle, 1.0)
            image_of_rotatsleeve = cv2.warpAffine(image, rotated_image_matrix, (row, col))
            image_sleeve = image_of_rotatsleeve[(offsetted_face[1]+offsetted_face[3])*(1-a):(offsetted_face[1]+5*offsetted_face[3])*(1+a),
                                (offsetted_face[0]-2*offsetted_face[2]))*(1-a):(offsetted_face[0]+3*offsetted_face[2])*(1+a)]
            if np.array(image_sleeve.shape).all() > 0:
                resized_image_sleeve = cv2.resize(image_sleeve, output_images_size)
                image_call = str(image_number)
                ###########################################################
                # finding which type sleeve is it, and its designation.
                # 1 - no_sleeve, 2, - short_sleeve, 3 - long_sleeve
                if 'no_sleeve' in type:
                    image_call = image_call + '_1'
                elif 'short_sleeve' in type:
                    image_call = image_call + '_2'
                elif 'long_sleeve' in type:
                    image_call = image_call + '_3'
                # ###########################################################
                resized_image_sleeve = Image.fromarray(whiten_image(resized_image_sleeve))
                # enhance the image:
                enhancers = []
                enhancers.append(ImageEnhance.Sharpness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Brightness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Contrast(resized_image_sleeve))
                enhancers.append(ImageEnhance.Color(resized_image_sleeve))
                for enhancer in enhancers:
                    for factor in enhancement_factors:
                        image_number += 1
                        resized_image_sleeve = enhancer.enhance(factor)
                        cv2.imwrite(directory_path + '/' + str(image_number) + '_' + image_call + image_file_name[-4:], PIL2array(resized_image_sleeve))

            # ###########################################################
            # writing a false, i.e. [0, 0, 0, 0, 0]:
            image_sleeve = image_of_rotatsleeve[(offsetted_face[1]+3*offsetted_face[3])*(1-a)+offsetted_face[3]:
                                (offsetted_face[1]+2.5*offsetted_face[3])*(1+a)+offsetted_face[3],
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            if np.array(image_sleeve.shape).all() > 0:
                resized_image_sleeve = cv2.resize(image_sleeve, output_images_size)
                image_call = str(image_number) + '_0'
                resized_image_sleeve = Image.fromarray(whiten_image(resized_image_sleeve))
                # enhance the image:
                enhancers = []
                enhancers.append(ImageEnhance.Sharpness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Brightness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Contrast(resized_image_sleeve))
                enhancers.append(ImageEnhance.Color(resized_image_sleeve))
                for enhancer in enhancers:
                    for factor in enhancement_factors:
                        for angle_for_zero in [0, 90, 180, 270]:
                            image_number += 1
                            resized_image_sleeve = enhancer.enhance(factor)
                            resized_image_sleeve_rotated = resized_image_sleeve.rotate(angle_for_zero)
                            cv2.imwrite(directory_path + '/' + str(image_number) + '_' + image_call + image_file_name[-4:], PIL2array(resized_image_sleeve_rotated))

        # flip along vertical axis:
        image = np.fliplr(image)
        face = face_cascade.detectMultiScale(image, 1.1, 2)
        # checking if the face (ancore) is present / detected:
        if len(face) == 0:
            continue
        face = face[0]
        offsetted_face = face
        row, col, dep = image.shape

        # no flip along vertical axis:sleeve_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
        flippsleeve_image_center_point = (col - (face[0]+0.5*face[2]), face[1]+1.5*face[3])
        # for offset1 in offset_range:
        #     offsetted_face[0] = face[0] #+ offset1 * face[2]
            # for offset2 in offset_range:
            # offsetted_face[1] = face[1] #+ offset2 * face[3]
        for angle in range(-max_angle, max_angle+1, angle_offset):
            rotated_image_matrix = cv2.getRotationMatrix2D(flippsleeve_image_center_point, angle, 1.0)
            image_of_rotatsleeve = cv2.warpAffine(image, rotated_image_matrix,(row, col))
            image_sleeve = image_of_rotatsleeve[(offsetted_face[1]+offsetted_face[3])*(1-a):(offsetted_face[1]+5*offsetted_face[3])*(1+a),
                                (offsetted_face[0]-2*offsetted_face[2]))*(1-a):(offsetted_face[0]+3*offsetted_face[2])*(1+a)]
            if np.array(image_sleeve.shape).all() > 0:
                resized_image_sleeve = cv2.resize(image_sleeve, output_images_size)
                image_call = str(image_number)
                ###########################################################
                # finding which type sleeve is it, and its designation.
                # 1 - no_sleeve, 2, - short_sleeve, 3 - long_sleeve
                if 'no_sleeve' in type:
                    image_call = image_call + '_1'
                elif 'short_sleeve' in type:
                    image_call = image_call + '_2'
                elif 'long_sleeve' in type:
                    image_call = image_call + '_3'
                # ###########################################################
                resized_image_sleeve = Image.fromarray(whiten_image(resized_image_sleeve))
                # enhance the image:
                enhancers = []
                enhancers.append(ImageEnhance.Sharpness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Brightness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Contrast(resized_image_sleeve))
                enhancers.append(ImageEnhance.Color(resized_image_sleeve))
                for enhancer in enhancers:
                    for factor in enhancement_factors:
                        image_number += 1
                        resized_image_sleeve = enhancer.enhance(factor)
                        cv2.imwrite(directory_path + '/' + str(image_number) + '_' + image_call + image_file_name[-4:], PIL2array(resized_image_sleeve))
            # ###########################################################
            # writing a false, i.e. [0, 0, 0, 0, 0]:
            image_sleeve = image_of_rotatsleeve[(offsetted_face[1]+1.5*offsetted_face[3])*(1-a)+offsetted_face[3]:
                                (offsetted_face[1]+2.5*offsetted_face[3])*(1+a)+offsetted_face[3],
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            if np.array(image_sleeve.shape).all() > 0:
                resized_image_sleeve = cv2.resize(image_sleeve, output_images_size)
                image_call = str(image_number) + '_0'
                resized_image_sleeve = Image.fromarray(whiten_image(resized_image_sleeve))
                # enhance the image:
                enhancers = []
                enhancers.append(ImageEnhance.Sharpness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Brightness(resized_image_sleeve))
                enhancers.append(ImageEnhance.Contrast(resized_image_sleeve))
                enhancers.append(ImageEnhance.Color(resized_image_sleeve))
                for enhancer in enhancers:
                    for factor in enhancement_factors:
                        for angle_for_zero in [0, 90, 180, 270]:
                            image_number += 1
                            resized_image_sleeve = enhancer.enhance(factor)
                            resized_image_sleeve_rotated = resized_image_sleeve.rotate(angle_for_zero)
                            cv2.imwrite(directory_path + '/' + str(image_number) + '_' + image_call + image_file_name[-4:], PIL2array(resized_image_sleeve_rotated))

            # print image_call
