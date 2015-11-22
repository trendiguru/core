import cv2
import numpy as np
# import scipy as sp
import os
# face_cascade = cv2.CascadeClassifier('/home/developer/python-packages/trendi/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('/home/nate/Desktop/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# current_directory = os.path.abspath()

image_file_types = ['.jpg','.png','.bmp','.gif']
a = 1.25 # scalar for increasing collar box in relation to face box (1==100%)
max_angle = 20 # tilt angle of the image for diversification
angle_offset = 10 # tilt angle of the image for diversification
# max_offset = 0.01 # maximum horizontal movement (% (out of box X) of the collar box for diversification
# delta_offset = max_offset # horizontal movement increments(%)
output_images_size = (32, 32) # pixels^2
dataset_directory_name = 'dataset'


current_directory_name = os.getcwd()
directory_path = current_directory_name + '/' + dataset_directory_name
if not os.path.exists(directory_path):
    os.mkdir(dataset_directory_name)

my_path = os.path.dirname(os.path.abspath(__file__))
only_files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f))]

only_image_files = []
for file_name in only_files:
    for image_type in image_file_types:
        if image_type in file_name:
            only_image_files.append(file_name)

# offset_range = np.arange(-max_offset, max_offset * 1.01, delta_offset)
# print offset_range
a = (a-1)/2
image_number = 0
for image_file_name in only_image_files:
    image = cv2.imread(image_file_name, 1)
    face = face_cascade.detectMultiScale(image, 1.3, 5)#[0]
    print image_file_name

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
                            (offsetted_face[1]+2.1*offsetted_face[3])*(1+a),
                            (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        image_call = str(image_number)
        ###########################################################
        # finding which type of collar is it, and its designation.
        # 1 - roundneck, 2 - squareneck, 3 - v-neck
        if ('crewneck' in image_file_name) or ('roundneck' in image_file_name) or \
                ('scoopneck' in image_file_name) or ('roundcollar' in image_file_name):
            image_call = image_call + '_1'
        #     data[collar_types].append('roundneck')
        #     data[collar_tag].append(1)
        #     data[collar_image].append(image_of_collar)
        elif 'squareneck' in image_file_name:
            image_call = image_call + '_2'
        #     data[collar_types].append('squareneck')
        #     data[collar_tag].append(2)
        #     data[collar_image].append(image_of_collar)
        elif 'v-neck' in image_file_name:
            image_call = image_call + '_3'
        #     data[collar_types].append('v-neck')
        #     data[collar_tag].append(3)
        #     data[collar_image].append(image_of_collar)
        # ###########################################################
        cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)
        # ###########################################################
        # cv2.imwrite(directory_path + '/' + image_file_name[0:-4] + '_x_offset' + str(offset1) + '_y_offset' +
        #             str(offset2) + '_offset_angle_' + str(angle) + '_unflipped' + image_file_name[-4:], image_of_collar)
        # cv2.imshow('cropped', image_of_collar)
        # cv2.waitKey(500)
        # ###########################################################
        # writing a false, i.e. [0, 0, 0]:
        image_of_collar = image_of_rotated_collar[(offsetted_face[1]+2.1*offsetted_face[3])*(1-a):
                            (offsetted_face[1]+3.1*offsetted_face[3])*(1+a),
                            (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        image_call = str(image_number) + '_0'
        cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)

    # flip along vertical axis:
    image = np.fliplr(image)
    for angle in range(-max_angle, max_angle+1, angle_offset):
        image_number += 1
        rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        image_of_collar = image_of_rotated_collar[(offsetted_face[1]+2.1*offsetted_face[3])*(1-a):
                            (offsetted_face[1]+3.1*offsetted_face[3])*(1+a),
                            col-((offsetted_face[0]+offsetted_face[2])*(1+a)):col-((offsetted_face[0])*(1-a))]
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        image_call = str(image_number)
        ###########################################################
        # finding which type of collar is it, and its designation.
        # 1 - roundneck, 2 - squareneck, 3 - v-neck
        if ('crewneck' in image_file_name) or ('roundneck' in image_file_name) or \
                ('scoopneck' in image_file_name) or ('roundcollar' in image_file_name):
            image_call += '_1'
        #     data[collar_types].append('roundneck')
        #     data[collar_tag].append(1)
        #     data[collar_image].append(image_of_collar)
        elif 'squareneck' in image_file_name:
            image_call += '_2'
        #     data[collar_types].append('squareneck')
        #     data[collar_tag].append(2)
        #     data[collar_image].append(image_of_collar)
        elif 'v-neck' in image_file_name:
            image_call += '_3'
        #     data[collar_types].append('v-neck')
        #     data[collar_tag].append(3)
        #     data[collar_image].append(image_of_collar)
        # ###########################################################
        cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)
        # ###########################################################
        # cv2.imwrite(directory_path + '/' + image_file_name[0:-4] + '_x_offset' + str(offset1) + '_y_offset' +
        #         str(offset2) + '_offset_angle_' + str(angle) + '_flipped' + image_file_name[-4:], image_of_collar)
        # cv2.imshow('cropped', image_of_collar)
        # cv2.waitKey(500)
        # ###########################################################
        # writing a false, i.e. [0, 0, 0]:
        image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                            (offsetted_face[1]+2.1*offsetted_face[3])*(1+a),
                            (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        image_call = str(image_number) + '_0'
        cv2.imwrite(directory_path + '/' + image_call + image_file_name[-4:], resized_image_of_collar)


print image_call
