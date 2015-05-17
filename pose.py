__author__ = 'Nadav Paz'

import numpy as np


def translate_2_boxes(boxes_list):
    """
    the function translates the boxes list to a boxes dictionary
    :param boxes_list: list of 1X106 float (0-103 are the boxes coordinates)
    :return:boxes dict: keys = body parts, values = boxes
    """
    for i in range(0, len(boxes_list)):
        boxes_list[i] = abs(float(boxes_list[i]))
    boxes_list = np.array(boxes_list)
    boxes_list = boxes_list.astype(np.uint16)
    boxes_dict = {'head': [], 'torso': [], 'left_arm': [], 'left_leg': [], 'right_arm': [], 'right_leg': []}
    for i in range(0, len(boxes_list) / 4):
        if i in [0, 1]:
            boxes_dict["head"].append(
                [boxes_list[4 * i], boxes_list[4 * i + 1], boxes_list[4 * i + 2], boxes_list[4 * i + 3]])
        elif i in [2, 7, 8, 9, 14, 19, 20, 21]:
            boxes_dict["torso"].append(
                [boxes_list[4 * i], boxes_list[4 * i + 1], boxes_list[4 * i + 2], boxes_list[4 * i + 3]])
        elif i in [3, 4, 5, 6]:
            boxes_dict["left_arm"].append(
                [boxes_list[4 * i], boxes_list[4 * i + 1], boxes_list[4 * i + 2], boxes_list[4 * i + 3]])
        elif i in [10, 11, 12, 13]:
            boxes_dict["left_leg"].append(
                [boxes_list[4 * i], boxes_list[4 * i + 1], boxes_list[4 * i + 2], boxes_list[4 * i + 3]])
        elif i in [15, 16, 17, 18]:
            boxes_dict["right_arm"].append(
                [boxes_list[4 * i], boxes_list[4 * i + 1], boxes_list[4 * i + 2], boxes_list[4 * i + 3]])
        elif i in [22, 23, 24, 25]:
            boxes_dict["right_leg"].append(
                [boxes_list[4 * i], boxes_list[4 * i + 1], boxes_list[4 * i + 2], boxes_list[4 * i + 3]])
    # image_head = image.copy()
    # image_left_arm = image.copy()
    # image_torso = image.copy()
    # image_left_leg = image.copy()
    # image_right_arm = image.copy()
    # image_right_leg = image.copy()
    #
    # for box in boxes_dict["head"]:
    # cv2.rectangle(image_head, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)
    # for box in boxes_dict["left_arm"]:
    # cv2.rectangle(image_left_arm, (box[0], box[1]), (box[2], box[3]), [190, 180, 255], 2)
    # for box in boxes_dict["torso"]:
    #     cv2.rectangle(image_torso, (box[0], box[1]), (box[2], box[3]), [0, 255, 255], 2)
    # for box in boxes_dict["left_leg"]:
    #     cv2.rectangle(image_left_leg, (box[0], box[1]), (box[2], box[3]), [0, 0, 255], 2)
    # for box in boxes_dict["right_arm"]:
    #     cv2.rectangle(image_right_arm, (box[0], box[1]), (box[2], box[3]), [255, 255, 0], 2)
    # for box in boxes_dict["right_leg"]:
    #     cv2.rectangle(image_right_leg, (box[0], box[1]), (box[2], box[3]), [255, 0, 0], 2)
    # cv2.imshow('head', image_head)
    # cv2.imshow('left arm', image_left_arm)
    # cv2.imshow('torso', image_torso)
    # cv2.imshow('left leg', image_left_leg)
    # cv2.imshow('right arm', image_right_arm)
    # cv2.imshow('right leg', image_right_leg)
    # cv2.waitKey(0)
    return boxes_dict


def pose_est_face(boxes_dict, image):
    box0 = boxes_dict["head"][0]
    box1 = boxes_dict["head"][1]
    box0_w = box0[2] - box0[0]
    box0_h = box0[3] - box0[1]
    box1_w = box1[2] - box1[0]
    box1_h = box1[3] - box1[1]
    avg_x0 = np.floor(np.mean([box0[0], box1[0]])).astype(np.uint16)
    avg_y0 = np.floor(np.mean([box0[1], box1[1]])).astype(np.uint16)
    avg_w = np.floor(np.mean([box0_w, box1_w])).astype(np.uint16)
    avg_h = np.floor(np.mean([box0_h, box1_h])).astype(np.uint16)
    new_face_rect = [avg_x0 + (0.05 * avg_w), avg_y0 + (0.05 * avg_h), 0.9 * avg_w, 0.9 * avg_h]
    # cv2.rectangle(image, (avg_x0, avg_y0), (avg_x0+avg_w, avg_y0+avg_h), [0, 255, 0], 2)
    # cv2.imshow('pose estimation face', image)
    # cv2.waitKey(0)
    return [new_face_rect]