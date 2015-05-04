__author__ = 'Nadav Paz'

import numpy as np


def translate_2_boxes(boxes_list):
    """
    the function translates the boxes list to a boxes dictionary
    :param boxes_list: np.array(uint16) of 1X106 float (0-103 are the boxes coordinates)
    :return:boxes dict: keys = body parts, values = boxes
    """

    for i in range(0, len(boxes_list)):
        boxes_list[i] = abs(float(boxes_list[i]))
    # boxes_list = np.array(boxes_list)
    # boxes_list = boxes_list.astype(np.uint16)
    boxes_dict = {'head': [], 'torso': [], 'left_arm': [], 'left_leg': [], 'right_arm': [], 'right_leg': []}
    for i in range(0, len(boxes_list)/4):
        if i in [0, 1]:
            boxes_dict["head"].append([boxes_list[4*i], boxes_list[4*i+1], boxes_list[4*i+2], boxes_list[4*i+3]])
        elif i in [2, 7, 8, 9, 14, 19, 20, 21]:
            boxes_dict["torso"].append([boxes_list[4*i], boxes_list[4*i+1], boxes_list[4*i+2], boxes_list[4*i+3]])
        elif i in [3, 4, 5, 6]:
            boxes_dict["left_arm"].append([boxes_list[4*i], boxes_list[4*i+1], boxes_list[4*i+2], boxes_list[4*i+3]])
        elif i in [10, 11, 12, 13]:
            boxes_dict["left_leg"].append([boxes_list[4*i], boxes_list[4*i+1], boxes_list[4*i+2], boxes_list[4*i+3]])
        elif i in [15, 16, 17, 18]:
            boxes_dict["right_arm"].append([boxes_list[4*i], boxes_list[4*i+1], boxes_list[4*i+2], boxes_list[4*i+3]])
        elif i in [22, 23, 24, 25]:
            boxes_dict["right_leg"].append([boxes_list[4*i], boxes_list[4*i+1], boxes_list[4*i+2], boxes_list[4*i+3]])

    return boxes_dict

"""
mateng = matlab.engine.start_matlab('-nodisplay')
# should be in the right dir = /home/ubuntu/Dev/pose_estimation/20121128-pose-release-ver1.3/code-basic/images...
image_path = 'home/ubuntu/Dev/pose_estimation/20121128-pose-release-ver1.3/code-basic/images/'
mateng.demo_func(image_path + filename)
"""
