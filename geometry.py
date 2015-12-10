__author__ = 'Nadav Paz'

import os

import cv2
import numpy as np

from . import kassper
from . import background_removal
from . import Utils


def higher_lower_body_split_line(face):
    w, y, w, h = face
    y_split = round(y + 3.6 * h)
    return y_split


def length_of_lower_body_part_field(image, face):
    """
    TODO
    """

    def legs_upper_line_cnt(mask):
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY, 0)
        contours = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[1]
        max_grade = 0
        line = mask.shape[0]
        for contour in contours:
            area = cv2.contourArea(contour)
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            moments = cv2.moments(contour)
            cy = int(moments['m01'] / moments['m00'])
            grade = 0.7 * cy + 0.3 * area
            if (topmost[1] > 5) and (bottommost[1] > 0.5 * mask.shape[0]):
                if grade > max_grade:
                    max_grade = grade
                    line = topmost[1]
        return int(line)
    image, rr = background_removal.standard_resize(image, 400)
    face = np.array([int(num) for num in face / rr], dtype=np.uint8)
    gc_image = background_removal.get_masked_image(image, background_removal.get_fg_mask(image))
    y_split = higher_lower_body_split_line(face)
    lower_bgr = image[y_split:gc_image.shape[0] - 1, :, :]
    try:
        only_skin_down = kassper.skin_detection_with_grabcut(lower_bgr, image, face, 'skin')
    except:
        print 'Problem with the grabcut'
        return 0.5, 0
    only_skin_mask = kassper.clutter_removal(only_skin_down, 100)
    l = legs_upper_line_cnt(255 * only_skin_mask) + int(y_split)
    if l > 6 * face[3]:
        return 1, l
    elif l < y_split:
        return 0, l
    else:
        return (l - y_split) / (face[1] + 6 * face[3] - y_split), l


def length_of_lower_body_db_dresses(image):
    """
    hello, this function will estimate and grade the length of the dress.
    dresses in the DB appear:
        1. with a woman inside - face clear
        2. with a woman inside - no face/ with face that wasn't found
        3. without woman inside
    for each one I have to find a solution that will satisfy them all equally.

    :param image: 3d ndarray
    :return:
    """


def collect_distances(dir, i):
    images = Utils.get_images_list(dir)[i * 10:10 * (i + 1)]
    print "Total {0} images".format(len(images))
    dist = []
    i = 0
    for image in images:
        face = background_removal.find_face_cascade(image)['faces'][0]
        print face
        if face is None:
            pass
        elif len(face) == 0:
            pass
        else:
            x, y, w, h = face
            while y + h < image.shape[0]:
                cv2.rectangle(image, (x, y), (x + w, y + h), [66, 0, 35], 2)
                y += h
            try:
                lod, line = length_of_lower_body_part_field(image, face)
                print lod
                cv2.line(image, (0, line), (image.shape[1], line), [0, 170, 170], 2)
                cv2.imwrite(os.getcwd() + '/' + str(i) + '.jpg', image)
            except:
                print "Problem with the length.."
        i += 1
    return