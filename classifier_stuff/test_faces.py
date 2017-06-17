__author__ = 'Nadav Paz'

import string
import collections
import os
import dlib
import cv2
import numpy as np
import rq
from . import constants
from . import Utils
from . import ccv_facedetector as ccv
from . import kassper
import time
from functools import partial
from trendi import background_removal

detector = dlib.get_frontal_face_detector()

def compare(img_arr):
    background_removal.find_face_ccv(image_arr, max_num_of_faces=100):
    background_removal.find_face_cascade(image, max_num_of_faces=10):
    background_removal.find_face_dlib_with_scores(image, max_num_of_faces=100):
    background_removal.choose_faces(image, faces_list, max_num_of_faces):
    background_removal.score_face(face, image):
    background_removal.face_is_relevant(image, face):
    background_removal.is_skin_color(face_ycrcb):
# variance_of_laplacian(image):
# is_one_color_image(image):
# average_bbs(bb1, bb2):
# combine_overlapping_rectangles(bb_list):
# body_estimation(image, face):
# bb_mask(image, bounding_box):
# paperdoll_item_mask(item_mask, bb):
# create_mask_for_gc(rectangles, image):
# create_arbitrary(image):
#
# face_skin_color_estimation(image, face_rect):
# person_isolation(image, face):
# check_skin_percentage(image):
