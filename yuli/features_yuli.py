__author__ = 'yuli'


from paperdoll import paperdoll_parse_enqueue


def translate_2_boxes(boxes_arr, image):
    """
    the function translates the boxes array to a boxes dictionary
    :param boxes_list: array of 1X106 float (0-103 are the boxes coordinates)
    :return:boxes dict: keys = body parts, values = boxes
    """
    boxes_list = list(boxes_arr)
    for i in range(0, len(boxes_list)):
        boxes_list[i] = abs(float(boxes_list[i]))
    boxes_list = np.array(boxes_list)
    boxes_list = boxes_list.astype(np.uint16)
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
    image_head = image.copy()
    image_left_arm = image.copy()
    image_torso = image.copy()
    image_left_leg = image.copy()
    image_right_arm = image.copy()
    image_right_leg = image.copy()

    for box in boxes_dict["head"]:
        cv2.rectangle(image_head, (box[0], box[1]), (box[2], box[3]), [0, 255, 0], 2)
    for box in boxes_dict["left_arm"]:
        cv2.rectangle(image_left_arm, (box[0], box[1]), (box[2], box[3]), [190, 180, 255], 2)
    for box in boxes_dict["torso"]:
        cv2.rectangle(image_torso, (box[0], box[1]), (box[2], box[3]), [0, 255, 255], 2)
    for box in boxes_dict["left_leg"]:
        cv2.rectangle(image_left_leg, (box[0], box[1]), (box[2], box[3]), [0, 0, 255], 2)
    for box in boxes_dict["right_arm"]:
        cv2.rectangle(image_right_arm, (box[0], box[1]), (box[2], box[3]), [255, 255, 0], 2)
    for box in boxes_dict["right_leg"]:
        cv2.rectangle(image_right_leg, (box[0], box[1]), (box[2], box[3]), [255, 0, 0], 2)
    cv2.imshow('head', image_head)
    cv2.imshow('left arm', image_left_arm)
    cv2.imshow('torso', image_torso)
    cv2.imshow('left leg', image_left_leg)
    cv2.imshow('right arm', image_right_arm)
    cv2.imshow('right leg', image_right_leg)
    cv2.waitKey(0)
    return boxes_dict

if __name__ == "__main__":
    url = '10796178-sexy-short-gown.jpg'
    img, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(url, async=False)

    box_dict = translate_2_boxes(pose, url)

# import background_removal
# import cv2
# import numpy as np
#
# img = cv2.imread('10796178-sexy-short-gown.jpg')
# rel = background_removal.image_is_relevant(img)
#
# facearr = np.asarray(rel[1][0])
#
# X = facearr[0]
# Y
# W
# H
#
# img = cv2.rectangle(img,(facearr[0],facearr[1]),(facearr[0]+facearr[2],facearr[1]+facearr[3]),(0,0,255) )
# img = cv2.rectangle(img,(facearr[0],facearr[1]+facearr[3]),(facearr[0]+facearr[2],facearr[1]+facearr[3]),(255,0,0) )
# img = cv2.rectangle(img,(facearr[0],facearr[1]),(facearr[0]+facearr[2],facearr[1]+facearr[3]),(0,255,0) )
#
#
# cv2.imshow('face',img)
# cv2.waitKey(0)
#
#
#
