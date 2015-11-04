__author__ = 'Nadav Paz'

import os

import cv2
import numpy as np

import kassper
import background_removal
import Utils


def higher_lower_body_split_line(face):
    box_height = face[3]
    y_split = face[1] + 4.5 * box_height
    return y_split


def length_of_lower_body_part_field(image, face):
    """
    TODO
    """

    def legs_upper_line_cnt(mask):
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY, 0)
        contours = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[1]
        y_up = mask.shape[0]
        topmost_list = []
        for contour in contours:
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            if topmost[1] > 5 and cv2.contourArea(contour) > 100:
                topmost_list.append(topmost[1])
        if len(topmost_list) > 0:
            y_up = np.amin(topmost_list)
        return int(y_up)

    # TODO - check if there are enough faces down the images..
    image, rr = background_removal.standard_resize(image, 400)
    face = np.array([int(num) for num in face / rr])
    gc_image = background_removal.get_masked_image(image, background_removal.get_fg_mask(image))
    y_split = higher_lower_body_split_line(face)
    lower_bgr = image[y_split:gc_image.shape[0] - 1, :, :]
    try:
        only_skin_down = kassper.skin_detection_with_grabcut(lower_bgr, image, 'skin')
    except:
        print 'Problem with the grabcut'
        return -1, -1
    only_skin_mask = kassper.clutter_removal(only_skin_down, 100)
    legs_up_cnt = legs_upper_line_cnt(255 * only_skin_mask) + int(y_split)
    return legs_up_cnt


def collect_distances(dir):
    images = Utils.get_images_list(dir)[:10]
    print "Total {0} images".format(len(images))
    dist = []
    for image in images:
        faces = background_removal.find_face_cascade(image)
        if faces is None:
            pass
        elif len(faces) == 0:
            pass
        else:
            line = length_of_lower_body_part_field(image, faces[0])
            cv2.line(image, (0, line), (image.shape[1], line), [0, 170, 170], 2)
            cv2.imwrite(os.getcwd() + images.index(image) + '.jpg', image)
            dist.append((line - faces[0][1]) / float(faces[0][3]))
            # print (line - faces[0][1]) / float(faces[0][3])
    # avrg = sum(dist) / float(len(dist))
    # stdev = statistics.stdev(dist)
    # print "Average is {0}, stdev is {1}".format(avrg, stdev)
    return dist
