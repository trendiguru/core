import cv2
import numpy as np

def collar_set_creator(image):

    collar_set = []
    face_cascade = cv2.CascadeClassifier('/home/nate/core/classifiers/haarcascade_frontalface_default.xml')
    a = 1.25 # scalar for increasing collar box in relation to face box (1==100%)
    max_angle = 5 # tilt angle of the image for diversification
    angle_offset = 5 # tilt angle of the image for diversification
    max_offset = 0.01 # maximum horizontal movement (% (out of box X) of the collar box for diversification
    delta_offset = max_offset # horizontal movement increments (%)
    output_images_size = (32, 32) # pixels^2

    offset_range = np.arange(-max_offset, max_offset * 1.01, delta_offset)
    a = (a-1)/2
    face = face_cascade.detectMultiScale(image, 1.3, 5)#[0]

    # checking if the face (ancore) is present / detected:
    if len(face) == 0:
        return collar_set
    face = face[0]
    offsetted_face = face
    row, col, dep = image.shape

    # no flip along vertical axis:
    collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
    image = np.fliplr(image)
    flipped_collar_image_center_point = (col - face[0]+0.5*face[2], face[1]+1.5*face[3])
    for offset1 in offset_range:
        offsetted_face[0] = face[0] + offset1 * face[2]
        for offset2 in offset_range:
            offsetted_face[1] = face[1] + offset2 * face[3]
            for angle in range(-max_angle, max_angle+1, angle_offset):
                image_number += 1
                rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
                image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
                image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                    (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                    (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                collar_set.append(resized_image_of_collar)

            # flip along vertical axis:
            for angle in range(-max_angle, max_angle+1, angle_offset):
                image_number += 1
                rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
                image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
                image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                    (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                    col-((offsetted_face[0]+offsetted_face[2])*(1+a)):col-((offsetted_face[0])*(1-a))]
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                collar_set.append(resized_image_of_collar)

    print collar_set
    return collar_set

collar_set_creator(cv2.imread('crewneck_17.png', 1))