import cv2
import numpy as np
# import scipy as sp
import os
face_cascade = cv2.CascadeClassifier('/home/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('/home/developer/python-packages/trendi/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('/home/nate/Desktop/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# current_directory = os.path.abspath()

image_file_types = ['.jpg', 'jpeg', '.png', '.bmp', '.gif']
a = 1.35 # scalar for increasing collar box in relation to face box (1==100%)
max_angle = 10 # tilt angle of the image for diversification
angle_offset = 5 # tilt angle of the image for diversification
# max_offset = 0.01 # maximum horizontal movement (% (out of box X) of the collar box for diversification
# delta_offset = max_offset # horizontal movement increments(%)
output_images_size = (32, 32) # pixels^2
dataset_directory_name = 'dataset'


current_directory_name = os.getcwd()
directory_path = current_directory_name + '/' + dataset_directory_name
if not os.path.exists(directory_path):
    os.mkdir(dataset_directory_name)

my_path = os.path.dirname(os.path.abspath(__file__)) + '/'
neck_type_images_directory = ['crewneck', 'roundneck', 'scoopneck', 'squareneck', 'vneck']
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

        # no flip along vertical axis:
        collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
        flipped_collar_image_center_point = (col - (face[0]+0.5*face[2]), face[1]+1.5*face[3])
        # for offset1 in offset_range:
        #     offsetted_face[0] = face[0] #+ offset1 * face[2]
            # for offset2 in offset_range:
            # offsetted_face[1] = face[1] #+ offset2 * face[3]
        for angle in range(-max_angle, max_angle+1, angle_offset):
            image_number += 1
            rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
            image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            if np.array(image_of_collar.shape).all() > 0:
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                image_call = str(image_number)
                ###########################################################
                # finding which type of collar is it, and its designation.
                # 1 - crewneck, 2, - roundneck, 3 - scoopneck, 4 - squareneck, 5 - v-neck
                if 'crewneck' in type:
                    image_call = image_call + '_1'
                elif 'roundneck' in type:
                    image_call = image_call + '_2'
                elif 'scoopneck' in type:
                    image_call = image_call + '_3'
                elif 'squareneck' in type:
                    image_call = image_call + '_4'
                elif 'vneck' in type:
                    image_call = image_call + '_5'
                # ###########################################################
                cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)
            # ###########################################################
            # writing a false, i.e. [0, 0, 0, 0, 0]:
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+1.5*offsetted_face[3])*(1-a)+offsetted_face[3]:
                                (offsetted_face[1]+2.5*offsetted_face[3])*(1+a)+offsetted_face[3],
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            if np.array(image_of_collar.shape).all() > 0:
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                image_call = str(image_number) + '_0'
                cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)

        # flip along vertical axis:
        image = np.fliplr(image)
        face = face_cascade.detectMultiScale(image, 1.1, 2)
        # checking if the face (ancore) is present / detected:
        if len(face) == 0:
            continue
        face = face[0]
        offsetted_face = face
        row, col, dep = image.shape

        # no flip along vertical axis:
        collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
        flipped_collar_image_center_point = (col - (face[0]+0.5*face[2]), face[1]+1.5*face[3])
        # for offset1 in offset_range:
        #     offsetted_face[0] = face[0] #+ offset1 * face[2]
            # for offset2 in offset_range:
            # offsetted_face[1] = face[1] #+ offset2 * face[3]
        for angle in range(-max_angle, max_angle+1, angle_offset):
            image_number += 1
            rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
            image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            if np.array(image_of_collar.shape).all() > 0:
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                image_call = str(image_number)
                ###########################################################
                # finding which type of collar is it, and its designation.
                # 1 - crewneck, 2, - roundneck, 3 - scoopneck, 4 - squareneck, 5 - v-neck
                if 'crewneck' in type:
                    image_call = image_call + '_1'
                elif 'roundneck' in type:
                    image_call = image_call + '_2'
                elif 'scoopneck' in type:
                    image_call = image_call + '_3'
                elif 'squareneck' in type:
                    image_call = image_call + '_5'
                elif 'vneck' in type:
                    image_call = image_call + '_5'
                # ###########################################################
                cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)
            # ###########################################################
            # writing a false, i.e. [0, 0, 0, 0, 0]:
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+1.5*offsetted_face[3])*(1-a)+offsetted_face[3]:
                                (offsetted_face[1]+2.5*offsetted_face[3])*(1+a)+offsetted_face[3],
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            if np.array(image_of_collar.shape).all() > 0:
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                image_call = str(image_number) + '_0'
                cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)


            # print image_call
