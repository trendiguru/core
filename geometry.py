__author__ = 'Nadav Paz'

import cv2
import numpy as np

import kassper
import background_removal


def item_length(image):
    """
    TODO
    """

    def higher_lower_body_split_line(face):
        box_width = face[2]
        y_split = face[1] + 4.5 * box_width
        return y_split

    def dress_length():
        lower_body_ycrcb = YCrCb_image[y_split:gc_image.shape[0] - 1, :, :]
        lower_bgr = cv2.cvtColor(lower_body_ycrcb, cv2.COLOR_YCR_CB2BGR)
        try:
            only_skin_down = kassper.skin_detection_with_grabcut(lower_bgr, image, 'skin')
        except:
            print 'Problem with the grabcut'
            return -1, -1
        only_skin_down = background_removal.get_masked_image(lower_bgr, kassper.clutter_removal(only_skin_down, 100))
        mask = kassper.get_mask(only_skin_down)
        legs_up_cnt = legs_upper_line_cnt(mask) + int(y_split)
        # legs_up = legs_upper_line(only_skin_down) + int(y_split)
        # cv2.line(image, (0, legs_up), (image.shape[1]-1, legs_up), [0, 255, 0], 3)
        cv2.line(image, (0, legs_up_cnt), (image.shape[1] - 1, legs_up_cnt), [255, 0, 255], 3)
        x, y, w, h = face
        for i in range(0, 8):
            cv2.rectangle(image, (x, y + i * h), (x + w, y + h * (1 + h)), [0, 255, 0], 1)
        cv2.imshow('only_skin_down', only_skin_down)
        cv2.imshow('lower', lower_bgr)
        cv2.imshow('legs', image)
        cv2.waitKey(0)
        return

    # def legs_upper_line(lower_body):
    # num_of_non_empty_cols = 0
    #     upper_y = 0
    #     lower_mask = kassper.get_mask(lower_body)
    #     for j in range(0, lower_mask.shape[1]):
    #         for i in range(0, lower_mask.shape[0]):
    #             if lower_mask[i][j] != 0 and i > 5:
    #                 num_of_non_empty_cols += 1
    #                 upper_y += i
    #                 break
    #     if upper_y == 0:
    #         return lower_mask.shape[0]
    #     else:
    #         return int(upper_y/num_of_non_empty_cols)

    def legs_upper_line_cnt(mask):
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        y_up = mask.shape[0]
        topmost_list = []
        for contour in contours:
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            if topmost[1] > 5 and cv2.contourArea(contour) > 100:
                topmost_list.append(topmost[1])
        if len(topmost_list) > 0:
            y_up = np.amin(topmost_list)
        return int(y_up)

    gc_image = background_removal.get_fg_mask(image)
    YCrCb_image = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
    faces = background_removal.image_is_relevant(image).faces
    if len(faces) > 0:
        for face in faces:
            y_split = higher_lower_body_split_line(face)
            dress_length()
    else:
        print 'no faces were detected'
        return -1, -1, -1
        # TODO: add general code that deals with that case
    return
