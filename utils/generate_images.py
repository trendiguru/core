import cv2
import numpy as np
# import scipy as sp
import os


def generate_images(img_arr, max_angle = 5,n_angles=10, max_offset_x = 100,n_offsets_x=1,  max_offset_y = 100, n_offsets_y=1,
                    noise_level=0.05,n_noises=1,do_mirror_lr=True,do_mirror_ud=False,output_dir='./')
    '''
    generates a bunch of slight variations of image by rotating, translating, noising
    total # images generated is n_angles*n_offsets_x*n_offsets_y*n_noises, these are done in nested loops
    if you don't want a particular xform set n_whatever = 1
    :param img_arr: image array to vary
    :param max_angle: rotation limit (degrees)
    :param n_angles: number of rotated images
    :param max_offset_x: x offset limit (pixels)
    :param n_offsets_x: number of x-offset images
    :param max_offset_y: y offset limit (pixels)
    :param n_offsets_y: number of y-offset images
    :param max_noise: max gaussian noise to add - 0->no noise, 1->max noise (avg 128)
    :param n_noises: number of noised images
    :param do_mirror_lr: work on orig and x-axis-flipped copy
    :param do_mirror_ud: work on orig and x-axis-flipped copy
    :param output_dir: dir to write output images
    :return:
    '''

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if n_angles <2:
        angles = [0]
    else:
        angles = np.arange(-max_angle, max_angle , max_angle*2 / (n_angles-1))
    if n_offsets_x <2:
        offsets_x = [0]
    else:
        offsets_x = np.arange(-max_offset_x, max_offset_x, max_offset_x*2/(n_offsets_x-1))
    if n_offsets_y <2:
        offsets_y = [0]
    else:
        offsets_y = np.arange(-max_offset_y, max_offset_y, max_offset_y*2/(n_offsets_y-1))

    print offset_range_angles
    height=img_arr.shape[0]
    width=img_arr.shape[1]
    center = [height/2,width/2]

    reflections=[img_arr]
    if do_mirror:
        fimg=img_arr.copy()
        mirror_image = cv2.flip(fimg,0)
        reflections.append(mirror_image)
    for reflection in reflections:
        for angle in offset_range:
            offsetted_face[0] = face[0] + offset1 * face[2]
            for offset_x in offset_range_x:
                offsetted_face[1] = face[1] + offset2 * face[3]
                for angle in range(-max_angle, max_angle+1, angle_offset):
                    image_number += 1
                    rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
                    image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
                    image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                        (offsetted_face[1]+2*offsetted_face[3])*(1+a),
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

                # flip along vertical axis:
                image = np.fliplr(image)
                for angle in range(-max_angle, max_angle+1, angle_offset):
                    image_number += 1
                    rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
                    image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
                    image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                        (offsetted_face[1]+2*offsetted_face[3])*(1+a),
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

        # print image_call