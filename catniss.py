__author__ = 'Nadav Paz'

import cv2
import matplotlib.pyplot as plt

import background_removal
import Utils
import kassper


def item_length(image, gc_image):
    """
    TODO
    """

    def higher_lower_body_split_line(face_rect):
        box_width = face_rect[0][2]
        y_split = face_rect[0][1] + 4.5 * box_width
        return y_split

    def dress_length():
        lower_body_ycrcb = YCrCb_image[y_split:gc_image.shape[0] - 1, :, :]
        lower_bgr = cv2.cvtColor(lower_body_ycrcb, cv2.COLOR_YCR_CB2BGR)
        only_skin_down = kassper.skin_detection_with_grabcut(lower_bgr, image, 'skin')
        only_skin_down = background_removal.get_masked_image(lower_bgr, kassper.clutter_removal(only_skin_down, 500))
        legs_prop = legs_face_proportion(face_rect[0], only_skin_down)
        return legs_prop

    def sleeve_length():
        """
        """
        x, y, w, h = face_rect[0]
        upper_body_ycrcb = YCrCb_image[0:y_split, :, :]
        upper_bgr = cv2.cvtColor(upper_body_ycrcb, cv2.COLOR_YCR_CB2BGR)
        only_skin_up = kassper.skin_detection_with_grabcut(upper_bgr, upper_bgr, 'skin')
        only_skin_up = background_removal.get_masked_image(upper_bgr, kassper.clutter_removal(only_skin_up, 500))
        right_hand = only_skin_up[y + int(1.3 * h): y_split, 1:int(x + w / 2), :]
        left_hand = only_skin_up[y + int(1.3 * h): y_split, int(x + w / 2): only_skin_up.shape[1] - 1, :]
        prop_left, prop_right = arms_face_proportions(face_rect[0], right_hand, left_hand)
        return prop_left, prop_right

    def arms_face_proportions(face_rect, right_hand, left_hand):
        num_of_left_hand_pixels = cv2.countNonZero(left_hand[:, :, 0])
        num_of_right_hand_pixels = cv2.countNonZero(right_hand[:, :, 0])
        prop_left = num_of_left_hand_pixels / face_rect[3]
        prop_right = num_of_right_hand_pixels / face_rect[3]
        return prop_left, prop_right

    def legs_face_proportion(face_rect, lower_bgr_skin):
        num_of_legs_pixels = cv2.countNonZero(lower_bgr_skin[:, :, 0])
        legs_prop = num_of_legs_pixels / face_rect[3]
        return legs_prop

    YCrCb_image = cv2.cvtColor(gc_image, cv2.COLOR_BGR2YCR_CB)
    face_rect = background_removal.find_face(image)
    if len(face_rect) > 0:
        y_split = higher_lower_body_split_line(face_rect)
        prop_left, prop_right = sleeve_length()
        legs_prop = dress_length()
    else:
        print 'no faces were detected'
        return -1, -1, -1
        # TODO: add general code that deals with that case
    return legs_prop, prop_left, prop_right


# dress length testing
def item_length_test(dir):
    print dir
    images_list = Utils.get_images_list(dir)
    legs_prop_list = []
    for image in images_list:
        fg_mask = background_removal.get_fg_mask(image)
        gc_image = background_removal.get_masked_image(image, fg_mask)
        legs_prop, prop_left, prop_right = item_length(image, gc_image)
        if legs_prop != -1:
            legs_prop_list.append(legs_prop)
        plt.hist(legs_prop_list)
        plt.title("proportion Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    return
