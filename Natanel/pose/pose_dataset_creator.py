import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import sys
import tables
import scipy
from scipy import ndimage
import scipy.io as sio
import json
import h5py

# from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.layers import merge, Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2, activity_l1l2
from keras.utils import np_utils

# import the necessary packages for SLIC (superpixel segmentation):
from skimage.segmentation import slic, felzenszwalb, join_segmentations, \
    mark_boundaries, quickshift, random_walker, relabel_sequential
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage import io
import argparse


def plot_image_skeleton_for_testing(image, joints_location_vector):


    # Right ankle
    cv2.line(image, (joints_location_vector[0, 0], joints_location_vector[0, 1]),
            (joints_location_vector[1, 0], joints_location_vector[1, 1]), (255, 0, 0),
            thickness=2, lineType=8, shift=0)
    # Right knee
    cv2.line(image, (joints_location_vector[1, 0], joints_location_vector[1, 1]),
            (joints_location_vector[2, 0], joints_location_vector[2, 1]), (255*2/3, 0, 0),
            thickness=2, lineType=8, shift=0)
    # Right hip
    cv2.line(image, (joints_location_vector[2, 0], joints_location_vector[2, 1]),
            (joints_location_vector[3, 0], joints_location_vector[3, 1]), (0, 0, 255),
            thickness=2, lineType=8, shift=0)
    # Left hip
    cv2.line(image, (joints_location_vector[3, 0], joints_location_vector[3, 1]),
            (joints_location_vector[4, 0], joints_location_vector[4, 1]), (0, 255*2/3, 0),
            thickness=2, lineType=8, shift=0)
    # Left knee
    cv2.line(image, (joints_location_vector[4, 0], joints_location_vector[4, 1]),
            (joints_location_vector[5, 0], joints_location_vector[5, 1]), (0, 255, 0),
            thickness=2, lineType=8, shift=0)
    # Left ankle
    #
    # Right wrist
    cv2.line(image, (joints_location_vector[6, 0], joints_location_vector[6, 1]),
            (joints_location_vector[7, 0], joints_location_vector[7, 1]), (255, 0, 0),
            thickness=2, lineType=8, shift=0)
    # Right elbow
    cv2.line(image, (joints_location_vector[7, 0], joints_location_vector[7, 1]),
            (joints_location_vector[8, 0], joints_location_vector[8, 1]), (255*2/3, 0, 0),
            thickness=2, lineType=8, shift=0)
    # Right shoulder
    cv2.line(image, (joints_location_vector[8, 0], joints_location_vector[8, 1]),
            (joints_location_vector[9, 0], joints_location_vector[9, 1]), (0, 0, 255),
            thickness=2, lineType=8, shift=0)
    # Left shoulder
    cv2.line(image, (joints_location_vector[9, 0], joints_location_vector[9, 1]),
            (joints_location_vector[10, 0], joints_location_vector[10, 1]), (0, 255*2/3, 0),
            thickness=2, lineType=8, shift=0)
    # Left elbow
    cv2.line(image, (joints_location_vector[10, 0], joints_location_vector[10, 1]),
            (joints_location_vector[11, 0], joints_location_vector[11, 1]), (0, 255, 0),
            thickness=2, lineType=8, shift=0)
    # Left wrist
    #
    # Neck
    cv2.line(image, (joints_location_vector[12, 0], joints_location_vector[12, 1]),
            (joints_location_vector[13, 0], joints_location_vector[13, 1]), (0, 0, 255),
            thickness=2, lineType=8, shift=0)
    # Head top

    # left hip to right shoulder:
    cv2.line(image, (joints_location_vector[3, 0], joints_location_vector[3, 1]),
            (joints_location_vector[8, 0], joints_location_vector[8, 1]), (0, 0, 255),
            thickness=2, lineType=8, shift=0)
    # right hip to left shoulder:
    cv2.line(image, (joints_location_vector[2, 0], joints_location_vector[2, 1]),
            (joints_location_vector[9, 0], joints_location_vector[9, 1]), (0, 0, 255),
            thickness=2, lineType=8, shift=0)

    ## plot appearant joints as full circles:
    for i in range(len(joints_location_vector[:])):
        cv2.circle(image, (joints_location_vector[i, 0], joints_location_vector[i, 1]),
                   2, (0, 255/3, 255/2), thickness=joints_location_vector[i, 2]*2, lineType=8, shift=0)
        # print joints_location_vector[i, 2]

    # cv2.imshow('L', image)
    # cv2.waitKey(0)
    return image


def human_body_probability_map(image, joints_location_vector):

    # t = 2
    #calculating t (Pscale):
    upper_left_corner, lower_right_corner = pose_bbox(joints_location_vector)
    size = np.array(lower_right_corner) - np.array(upper_left_corner)
    Pscale = (max(size)*9) / (min(size)*2)
    # print 'Pscal = ' + str(Pscale)
    t = Pscale
    if t < 2:
        t = 2

    black_image = np.zeros(image.shape[:2], dtype='uint8')
    rbl = black_image.copy() # right bottom leg
    rul = black_image.copy() # right upper leg
    pelvic = black_image.copy() # pelvice
    lul = black_image.copy() # left upper leg
    lbl = black_image.copy() # left bottom leg
    rba = black_image.copy() # right bottom arm
    rua = black_image.copy() # right upper arm
    shoul = black_image.copy() # shoulders
    neck = black_image.copy() # neck
    lba = black_image.copy() # left bottom arm
    lua = black_image.copy() # left upper arm
    head = black_image.copy() # head
    ut = black_image.copy() # upper torso
    lt = black_image.copy() # lower torso
    ft = black_image.copy() # full torso
    td1 = black_image.copy()  # torso diagonal 1
    td2 = black_image.copy()  # torso diagonal 2

    rbl_p = ((joints_location_vector[0, 0], joints_location_vector[0, 1]),
             (joints_location_vector[1, 0], joints_location_vector[1, 1]))
    rul_p = ((joints_location_vector[1, 0], joints_location_vector[1, 1]),
             (joints_location_vector[2, 0], joints_location_vector[2, 1]))
    pelvic_p = ((joints_location_vector[2, 0], joints_location_vector[2, 1]),
                (joints_location_vector[3, 0], joints_location_vector[3, 1]))
    lul_p = ((joints_location_vector[3, 0], joints_location_vector[3, 1]),
             (joints_location_vector[4, 0], joints_location_vector[4, 1]))
    lbl_p = ((joints_location_vector[4, 0], joints_location_vector[4, 1]),
             (joints_location_vector[5, 0], joints_location_vector[5, 1]))
    rba_p = ((joints_location_vector[6, 0], joints_location_vector[6, 1]),
             (joints_location_vector[7, 0], joints_location_vector[7, 1]))
    rua_p = ((joints_location_vector[7, 0], joints_location_vector[7, 1]),
             (joints_location_vector[8, 0], joints_location_vector[8, 1]))
    shoul_p = ((joints_location_vector[8, 0], joints_location_vector[8, 1]),
               (joints_location_vector[9, 0], joints_location_vector[9, 1]))
    neck_p = ((sum(joints_location_vector[8:10, 0]) / 2, sum(joints_location_vector[8:10, 1]) / 2),
              (joints_location_vector[12, 0], joints_location_vector[12, 1]))
    lba_p = ((joints_location_vector[10, 0], joints_location_vector[10, 1]),
             (joints_location_vector[11, 0], joints_location_vector[11, 1]))
    lua_p = ((joints_location_vector[9, 0], joints_location_vector[9, 1]),
             (joints_location_vector[10, 0], joints_location_vector[10, 1]))
    head_p = ((joints_location_vector[12, 0], joints_location_vector[12, 1]),
              (joints_location_vector[13, 0], joints_location_vector[13, 1]))

    p4 = np.array((joints_location_vector[3, 0], joints_location_vector[3, 1]))
    p3 = np.array((joints_location_vector[8, 0], joints_location_vector[8, 1]))
    p1 = np.array((joints_location_vector[2, 0], joints_location_vector[2, 1]))
    p2 = np.array((joints_location_vector[9, 0], joints_location_vector[9, 1]))
    a1 = 1.0 * (p2[1] - p1[1]) / (p2[0] - p1[0])
    a2 = 1.0 * (p4[1] - p3[1]) / (p4[0] - p3[0])
    x_cross = 1.0 * (-a2 * (p3[0] - p1[0]) + p3[1] - p1[1]) / (a1 - a2)
    p5 = (int(p1[0] + x_cross), int(p1[1] + x_cross * a1))
    ut_p = np.array((p2, p3, p5), dtype='int32')
    lt_p = np.array((p1, p4, p5), dtype='int32')
    ft_p = np.array((p1, p3, p2, p4), dtype='int32')
    td1_p = ((joints_location_vector[3, 0], joints_location_vector[3, 1]),
             (joints_location_vector[8, 0], joints_location_vector[8, 1]))
    td2_p = ((joints_location_vector[2, 0], joints_location_vector[2, 1]),
             (joints_location_vector[9, 0], joints_location_vector[9, 1]))

    # Right ankle
    cv2.line(rbl, rbl_p[0], rbl_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[0:2, 1]):min(joints_location_vector[0:2, 1])+2*(max(joints_location_vector[0:2, 1])-min(joints_location_vector[0:2, 1])),
    #               min(joints_location_vector[0:2, 0]):min(joints_location_vector[0:2, 0])+2*(max(joints_location_vector[0:2, 0])-min(joints_location_vector[0:2, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[rbl==0] = 0
    # rbl = (255 * distance_mask).astype('uint8')
    # cv2.normalize(rbl, rbl, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([rbl, distance_mask]))
    # # cv2.waitKey(0)

    # Right knee
    cv2.line(rul, rul_p[0], rul_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[1:3, 1]):min(joints_location_vector[1:3, 1])+2*(max(joints_location_vector[1:3, 1])-min(joints_location_vector[1:3, 1])),
    #               min(joints_location_vector[1:3, 0]):min(joints_location_vector[1:3, 0])+2*(max(joints_location_vector[1:3, 0])-min(joints_location_vector[1:3, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[rul==0] = 0
    # rul = (255 * distance_mask).astype('uint8')
    # cv2.normalize(rul, rul, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([rul, distance_mask]))
    # # cv2.waitKey(0)

    # Right hip
    cv2.line(pelvic, pelvic_p[0], pelvic_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[2:4, 1]):min(joints_location_vector[2:4, 1])+2*(max(joints_location_vector[2:4, 1])-min(joints_location_vector[2:4, 1])),
    #               min(joints_location_vector[2:4, 0]):min(joints_location_vector[2:4, 0])+2*(max(joints_location_vector[2:4, 0])-min(joints_location_vector[2:4, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[pelvic==0] = 0
    # pelvic = (255 * distance_mask).astype('uint8')
    # cv2.normalize(pelvic, pelvic, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([pelvic, distance_mask]))
    # # cv2.waitKey(0)

    # Left hip
    cv2.line(lul, lul_p[0], lul_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[3:5, 1]):min(joints_location_vector[3:5, 1])+2*(max(joints_location_vector[3:5, 1])-min(joints_location_vector[3:5, 1])),
    #               min(joints_location_vector[3:5, 0]):min(joints_location_vector[3:5, 0])+2*(max(joints_location_vector[3:5, 0])-min(joints_location_vector[3:5, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[lul==0] = 0
    # lul = (255 * distance_mask).astype('uint8')
    # cv2.normalize(lul, lul, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([lul, distance_mask]))
    # # cv2.waitKey(0)

    # Left knee
    cv2.line(lbl, lbl_p[0], lbl_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[4:6, 1]):min(joints_location_vector[4:6, 1])+2*(max(joints_location_vector[4:6, 1])-min(joints_location_vector[4:6, 1])),
    #               min(joints_location_vector[4:6, 0]):min(joints_location_vector[4:6, 0])+2*(max(joints_location_vector[4:6, 0])-min(joints_location_vector[4:6, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[lbl==0] = 0
    # lbl = (255 * distance_mask).astype('uint8')
    # cv2.normalize(lbl, lbl, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([lbl, distance_mask]))
    # # cv2.waitKey(0)

    # Left ankle
    #
    # Right wrist
    cv2.line(rba, rba_p[0], rba_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[6:8, 1]):min(joints_location_vector[6:8, 1])+2*(max(joints_location_vector[6:8, 1])-min(joints_location_vector[6:8, 1])),
    #               min(joints_location_vector[6:8, 0]):min(joints_location_vector[6:8, 0])+2*(max(joints_location_vector[6:8, 0])-min(joints_location_vector[6:8, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[rba==0] = 0
    # rba = (255 * distance_mask).astype('uint8')
    # cv2.normalize(rba, rba, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([rba, distance_mask]))
    # # cv2.waitKey(0)

    # Right elbow
    cv2.line(rua, rua_p[0], rua_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[7:9, 1]):min(joints_location_vector[7:9, 1])+2*(max(joints_location_vector[7:9, 1])-min(joints_location_vector[7:9, 1])),
    #               min(joints_location_vector[7:9, 0]):min(joints_location_vector[7:9, 0])+2*(max(joints_location_vector[7:9, 0])-min(joints_location_vector[7:9, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[rua==0] = 0
    # rua = (255 * distance_mask).astype('uint8')
    # cv2.normalize(rua, rua, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([rua, distance_mask]))
    # # cv2.waitKey(0)

    # Right shoulder
    cv2.line(shoul, shoul_p[0], shoul_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[8:10, 1]):min(joints_location_vector[8:10, 1])+2*(max(joints_location_vector[8:10, 1])-min(joints_location_vector[8:10, 1])),
    #               min(joints_location_vector[8:10, 0]):min(joints_location_vector[8:10, 0])+2*(max(joints_location_vector[8:10, 0])-min(joints_location_vector[8:10, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[shoul==0] = 0
    # rua = (255 * distance_mask).astype('uint8')
    # cv2.normalize(shoul, shoul, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([shoul, distance_mask]))
    # # cv2.waitKey(0)

    # adding a neck:
    cv2.line(neck, neck_p[0], neck_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(sum(joints_location_vector[8:10, 1])/2, joints_location_vector[12, 1]):min(sum(joints_location_vector[8:10, 1])/2, joints_location_vector[12, 1])+2*(max(sum(joints_location_vector[8:10, 1])/2, joints_location_vector[12, 1])-min(sum(joints_location_vector[8:10, 1])/2, joints_location_vector[12, 1])),
    #               min(sum(joints_location_vector[8:10, 0])/2, joints_location_vector[12, 0]):min(sum(joints_location_vector[8:10, 0])/2, joints_location_vector[12, 0])+2*(max(sum(joints_location_vector[8:10, 0])/2, joints_location_vector[12, 0])-min(sum(joints_location_vector[8:10, 0])/2, joints_location_vector[12, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[neck==0] = 0
    # neck = (255 * distance_mask).astype('uint8')
    # cv2.normalize(neck, neck, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([neck, distance_mask]))
    # # cv2.waitKey(0)

    # Left shoulder
    cv2.line(lua, lua_p[0], lua_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[9:11, 1]):min(joints_location_vector[9:11, 1])+2*(max(joints_location_vector[9:11, 1])-min(joints_location_vector[9:11, 1])),
    #               min(joints_location_vector[9:11, 0]):min(joints_location_vector[9:11, 0])+2*(max(joints_location_vector[9:11, 0])-min(joints_location_vector[9:11, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[lua==0] = 0
    # lua = (255 * distance_mask).astype('uint8')
    # cv2.normalize(lua, lua, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([lua, distance_mask]))
    # # cv2.waitKey(0)

    # Left elbow
    cv2.line(lba, lba_p[0], lba_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[10:12, 1]):min(joints_location_vector[10:12, 1])+2*(max(joints_location_vector[10:12, 1])-min(joints_location_vector[10:12, 1])),
    #               min(joints_location_vector[10:12, 0]):min(joints_location_vector[10:12, 0])+2*(max(joints_location_vector[10:12, 0])-min(joints_location_vector[10:12, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[lba==0] = 0
    # lba = (255 * distance_mask).astype('uint8')
    # cv2.normalize(lba, lba, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([lba, distance_mask]))
    # # cv2.waitKey(0)

    # Left wrist
    #
    # Neck
    cv2.line(head, head_p[0], head_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[12:14, 1]):min(joints_location_vector[12:14, 1])+2*(max(joints_location_vector[12:14, 1])-min(joints_location_vector[12:14, 1])),
    #               min(joints_location_vector[12:14, 0]):min(joints_location_vector[12:14, 0])+2*(max(joints_location_vector[12:14, 0])-min(joints_location_vector[12:14, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[head==0] = 0
    # head = (255 * distance_mask).astype('uint8')
    # cv2.normalize(head, head, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([head, distance_mask]))
    # # cv2.waitKey(0)

    # Head top

    # left hip to right shoulder:
    cv2.line(td1, td1_p[0], td1_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[3, 1], joints_location_vector[8, 1]):min(joints_location_vector[3, 1], joints_location_vector[8, 1])+2*(max(joints_location_vector[3, 1], joints_location_vector[8, 1])-min(joints_location_vector[3, 1], joints_location_vector[8, 1])),
    #               min(joints_location_vector[3, 0], joints_location_vector[8, 0]):min(joints_location_vector[3, 0], joints_location_vector[8, 0])+2*(max(joints_location_vector[3, 0], joints_location_vector[8, 0])-min(joints_location_vector[3, 0], joints_location_vector[8, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[td1==0] = 0
    # td2 = (255 * distance_mask).astype('uint8')
    # cv2.normalize(td1, td1, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([td1, distance_mask]))
    # # cv2.waitKey(0)

    # right hip to left shoulder:
    cv2.line(td2, td2_p[0], td2_p[1], (255), thickness=t, lineType=8, shift=0)
    # distance_mask = black_image.copy()
    # distance_mask[min(joints_location_vector[2, 1], joints_location_vector[9, 1]):min(joints_location_vector[2, 1], joints_location_vector[9, 1])+2*(max(joints_location_vector[2, 1], joints_location_vector[9, 1])-min(joints_location_vector[2, 1], joints_location_vector[9, 1])),
    #               min(joints_location_vector[2, 0], joints_location_vector[9, 0]):min(joints_location_vector[2, 0], joints_location_vector[9, 0])+2*(max(joints_location_vector[2, 0], joints_location_vector[9, 0])-min(joints_location_vector[2, 0], joints_location_vector[9, 0]))] = 255
    # distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    # cv2.normalize(distance_mask, distance_mask, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[td2==0] = 0
    # td2 = (255 * distance_mask).astype('uint8')
    # cv2.normalize(td2, td2, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', np.hstack([td2, distance_mask]))
    # # cv2.waitKey(0)

    # fill body:

    # lower torso
    cv2.fillConvexPoly(lt, lt_p, (255), lineType=0, shift=0)
    # distance_mask = cv2.distanceTransform(lt, cv2.DIST_L2, 3)
    # cv2.normalize(lt, lt, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[lt == 0] = 0
    # lt = (255 * distance_mask).astype('uint8')
    # cv2.normalize(lt, lt, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', lt)
    # # cv2.waitKey(0)

    # upper torso
    cv2.fillConvexPoly(ut, ut_p, (255), lineType=0, shift=0)
    # distance_mask = cv2.distanceTransform(ut, cv2.DIST_L2, 3)
    # cv2.normalize(ut, ut, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[ut == 0] = 0
    # ut = (255 * distance_mask).astype('uint8')
    # cv2.normalize(ut, ut, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', ut)
    # # cv2.waitKey(0)

    # full torso
    cv2.fillConvexPoly(ft, ft_p, (255), lineType=0, shift=0)
    # distance_mask = cv2.distanceTransform(ut, cv2.DIST_L2, 3)
    # cv2.normalize(ft, ft, 0, 1.0, cv2.NORM_MINMAX)
    # distance_mask[ft == 0] = 0
    # ft = (255 * distance_mask).astype('uint8')
    # cv2.normalize(ft, ft, 0, 255, cv2.NORM_MINMAX)
    # # cv2.imshow('S', ft)
    # # cv2.waitKey(0)

    whole_body = rbl + rul + pelvic + lul + lbl + rba + rua + shoul + neck + lba + lua + head + ut + lt + ft + td1 + td2
    # whole_body =
    # rbl
    # rul
    # pelvic
    # lul
    # lbl
    # rba
    # rua
    # shoul
    # neck
    # lba
    # lua
    # head
    # ut
    # lt
    # ft
    # td1
    # td2

    # cv2.imshow('P', whole_body)
    # cv2.waitKey(0)

    limb_masks = [whole_body, rbl, rul, pelvic, lul, lbl, rba, rua, shoul, neck, lba, lua, head, td1, td2, ut, lt, ft]

    # # for show:
    # i = 0
    # for t in limb_masks:
    #     cv2.imshow('l', t)
    #     cv2.waitKey(0)
    #     print i
    #     i += 1

    return limb_masks


def spatiogram_mask_of_human_parts(image, joints_location_vector):

    rbl_p = ((joints_location_vector[0, 0], joints_location_vector[0, 1]),
             (joints_location_vector[1, 0], joints_location_vector[1, 1]))
    rul_p = ((joints_location_vector[1, 0], joints_location_vector[1, 1]),
             (joints_location_vector[2, 0], joints_location_vector[2, 1]))
    pelvic_p = ((joints_location_vector[2, 0], joints_location_vector[2, 1]),
                (joints_location_vector[3, 0], joints_location_vector[3, 1]))
    lul_p = ((joints_location_vector[4, 0], joints_location_vector[4, 1]),
             (joints_location_vector[3, 0], joints_location_vector[3, 1]))
    lbl_p = ((joints_location_vector[5, 0], joints_location_vector[5, 1]),
             (joints_location_vector[4, 0], joints_location_vector[4, 1]))
    rba_p = ((joints_location_vector[6, 0], joints_location_vector[6, 1]),
             (joints_location_vector[7, 0], joints_location_vector[7, 1]))
    rua_p = ((joints_location_vector[7, 0], joints_location_vector[7, 1]),
             (joints_location_vector[8, 0], joints_location_vector[8, 1]))
    shoul_p = ((joints_location_vector[8, 0], joints_location_vector[8, 1]),
               (joints_location_vector[9, 0], joints_location_vector[9, 1]))
    neck_p = ((sum(joints_location_vector[8:10, 0]) / 2, sum(joints_location_vector[8:10, 1]) / 2),
              (joints_location_vector[12, 0], joints_location_vector[12, 1]))
    lba_p = ((joints_location_vector[11, 0], joints_location_vector[11, 1]),
             (joints_location_vector[10, 0], joints_location_vector[10, 1]))
    lua_p = ((joints_location_vector[10, 0], joints_location_vector[10, 1]),
             (joints_location_vector[9, 0], joints_location_vector[9, 1]))
    head_p = ((joints_location_vector[12, 0], joints_location_vector[12, 1]),
              (joints_location_vector[13, 0], joints_location_vector[13, 1]))

    p4 = np.array((joints_location_vector[3, 0], joints_location_vector[3, 1]))
    p3 = np.array((joints_location_vector[8, 0], joints_location_vector[8, 1]))
    p1 = np.array((joints_location_vector[2, 0], joints_location_vector[2, 1]))
    p2 = np.array((joints_location_vector[9, 0], joints_location_vector[9, 1]))
    a1 = 1.0 * (p2[1] - p1[1]) / (p2[0] - p1[0])
    a2 = 1.0 * (p4[1] - p3[1]) / (p4[0] - p3[0])
    x_cross = 1.0 * (-a2 * (p3[0] - p1[0]) + p3[1] - p1[1]) / (a1 - a2)
    p5 = (int(p1[0] + x_cross), int(p1[1] + x_cross * a1))
    ut_p = np.array((p2, p3, p5), dtype='int32')
    lt_p = np.array((p1, p4, p5), dtype='int32')
    ft_p = np.array((p1, p3, p2, p4), dtype='int32')
    td1_p = ((joints_location_vector[3, 0], joints_location_vector[3, 1]),
             (joints_location_vector[8, 0], joints_location_vector[8, 1]))
    td2_p = ((joints_location_vector[2, 0], joints_location_vector[2, 1]),
             (joints_location_vector[9, 0], joints_location_vector[9, 1]))

    # rbl # right bottom leg
    # rul # right upper leg
    # pelvic # pelvice
    # lul # left upper leg
    # lbl # left bottom leg
    # rba # right bottom arm
    # rua  # right upper arm
    # shoul # shoulders
    # neck # neck
    # lba # left bottom arm
    # lua # left upper arm
    # head # head
    # ut # upper torso
    # lt # lower torso
    # ft # full torso
    # td1 # torso diagonal 1
    # td2 # torso diagonal 2


    whole_body, rbl, rul, pelvic, lul, lbl, rba, rua, shoul, neck, lba, lua, head, td1, td2, ut, lt, ft = \
        human_body_probability_map(image, joints_location_vector)

    ceieve = [rbl, rul, pelvic, lul, lbl, rba, rua, shoul, neck, lba, lua, head, td1, td2]
    ceieve_p = [rbl_p, rul_p, pelvic_p, lul_p, lbl_p, rba_p, rua_p, shoul_p, neck_p, lba_p, lua_p, head_p, td1_p, td2_p]

    t = 3 # bbox padding addition
    black_map = np.zeros(whole_body.shape, dtype='uint8')
    limb_spatiogram_masks = []

    # full body spatiogram map:
    whole_body = cv2.distanceTransform(whole_body, cv2.DIST_L2, 3)
    cv2.normalize(whole_body, whole_body, 0, 1.0, cv2.NORM_MINMAX)
    whole_body = (255 * (whole_body ** 2))
    limb_spatiogram_masks.append(whole_body)

    # body limbs spatiogram maps:
    for i in range(len(ceieve)):
        mask = ceieve[i]
        p = ceieve_p[i]
        map = black_map.copy()
        p1 = (min(p[0][1], p[1][1]), min(p[0][0], p[1][0]))
        p2 = (max(p[0][1], p[1][1]), max(p[0][0], p[1][0]))
        map[p1[0] - t:p2[0] + t, p1[1] - t:p2[1] + t] = 255
        # # TODO: fix zero point location!
        map[p[0][1], p[0][0]] = 0
        bbox = map[p1[0] - t:p2[0] + t, p1[1] - t:p2[1] + t]
        bbox = cv2.distanceTransform(bbox, cv2.DIST_L2, 3)
        cv2.normalize(bbox, bbox, 0, 1.0, cv2.NORM_MINMAX)
        bbox = (255 * bbox**2).astype('uint8')
        # cv2.normalize(bbox, bbox, 0, 255, cv2.NORM_MINMAX)
        map[p1[0] - t:p2[0] + t, p1[1] - t:p2[1] + t] = bbox
        mask[mask>0] = map[mask>0]
        limb_spatiogram_masks.append((255 * mask).astype('uint8'))

    # body polygons (torso) spatiogram maps:
    polygons = [ut, lt, ft]
    for poly in polygons:
        poly = cv2.distanceTransform(poly, cv2.DIST_L2, 3)
        cv2.normalize(poly, poly, 0, 1.0, cv2.NORM_MINMAX)
        poly = (255 * (poly ** 2))
        limb_spatiogram_masks.append(poly)

    p = np.sum(limb_spatiogram_masks[1:], 0).astype('uint8')
    # cv2.normalize(p, p, 0, 1., cv2.NORM_MINMAX)
    limb_spatiogram_masks.append(p)

    # # for show:
    # for t in limb_spatiogram_masks:
    #     cv2.imshow('l', t)
    #     cv2.waitKey(0)

    return limb_spatiogram_masks


def superpixel_masks_parsing(image, mask):

    # bbox for speed slicing:
    y0, x0, dy, dx = cv2.boundingRect(mask)
    number_of_segments = (dx*dy)/((dx+dy)/4)
    print number_of_segments
    image2 = (image.copy() * mask[:, :, np.newaxis])[x0:x0+dx, y0:y0+dy]

    # # apply SLIC and extract (approximately) the supplied number of segments:
    # segments = slic(image2, n_segments=number_of_segments, compactness=13, max_iter=13,
    #                 sigma=0.1, spacing=None, multichannel=True,
    #                 convert2lab=True, ratio=None)

    # apply QUICKSHIFT and extract (approximately) the supplied number of segments:
    segments = quickshift(image2, max_dist=np.sqrt(number_of_segments), sigma=0.1, convert2lab=True)

    segments = segments + 1 # So that no labelled region is 0 and ignored by regionprops
    segments = segments * mask[x0:x0+dx, y0:y0+dy]
    superpixel_msk = mask
    superpixel_msk[x0:x0+dx, y0:y0+dy] = segments

    return superpixel_msk


def color_reagion_mask(image, mask):

    superpixel_mask = superpixel_masks_parsing(image.copy(), mask)
    parsed_image = image.copy()
    print parsed_image
    for i in range(1, superpixel_mask.max()+1):
        average_color = np.zeros(3)
        for channel in range(3):
            average_color[channel] = np.mean(image[:, :, channel][superpixel_mask == i])
        parsed_image[superpixel_mask == i] = average_color.astype('uint8')

    return parsed_image



#TODO:############################################################
def weak_logical_segmentation(image, joints_location_vector):


    rbl_p = ((joints_location_vector[0, 0], joints_location_vector[0, 1]),
             (joints_location_vector[1, 0], joints_location_vector[1, 1]))
    rul_p = ((joints_location_vector[1, 0], joints_location_vector[1, 1]),
             (joints_location_vector[2, 0], joints_location_vector[2, 1]))
    pelvic_p = ((joints_location_vector[2, 0], joints_location_vector[2, 1]),
                (joints_location_vector[3, 0], joints_location_vector[3, 1]))
    lul_p = ((joints_location_vector[3, 0], joints_location_vector[3, 1]),
             (joints_location_vector[4, 0], joints_location_vector[4, 1]))
    lbl_p = ((joints_location_vector[4, 0], joints_location_vector[4, 1]),
             (joints_location_vector[5, 0], joints_location_vector[5, 1]))
    rba_p = ((joints_location_vector[6, 0], joints_location_vector[6, 1]),
             (joints_location_vector[7, 0], joints_location_vector[7, 1]))
    rua_p = ((joints_location_vector[7, 0], joints_location_vector[7, 1]),
             (joints_location_vector[8, 0], joints_location_vector[8, 1]))
    shoul_p = ((joints_location_vector[8, 0], joints_location_vector[8, 1]),
               (joints_location_vector[9, 0], joints_location_vector[9, 1]))
    neck_p = ((sum(joints_location_vector[8:10, 0]) / 2, sum(joints_location_vector[8:10, 1]) / 2),
              (joints_location_vector[12, 0], joints_location_vector[12, 1]))
    lba_p = ((joints_location_vector[10, 0], joints_location_vector[10, 1]),
             (joints_location_vector[11, 0], joints_location_vector[11, 1]))
    lua_p = ((joints_location_vector[9, 0], joints_location_vector[9, 1]),
             (joints_location_vector[10, 0], joints_location_vector[10, 1]))
    head_p = ((joints_location_vector[12, 0], joints_location_vector[12, 1]),
              (joints_location_vector[13, 0], joints_location_vector[13, 1]))

    p4 = np.array((joints_location_vector[3, 0], joints_location_vector[3, 1]))
    p3 = np.array((joints_location_vector[8, 0], joints_location_vector[8, 1]))
    p1 = np.array((joints_location_vector[2, 0], joints_location_vector[2, 1]))
    p2 = np.array((joints_location_vector[9, 0], joints_location_vector[9, 1]))
    a1 = 1.0 * (p2[1] - p1[1]) / (p2[0] - p1[0])
    a2 = 1.0 * (p4[1] - p3[1]) / (p4[0] - p3[0])
    x_cross = 1.0 * (-a2 * (p3[0] - p1[0]) + p3[1] - p1[1]) / (a1 - a2)
    p5 = (int(p1[0] + x_cross), int(p1[1] + x_cross * a1))
    ut_p = np.array((p2, p3, p5), dtype='int32')
    lt_p = np.array((p1, p4, p5), dtype='int32')
    ft_p = np.array((p1, p3, p2, p4), dtype='int32')
    td1_p = ((joints_location_vector[3, 0], joints_location_vector[3, 1]),
             (joints_location_vector[8, 0], joints_location_vector[8, 1]))
    td2_p = ((joints_location_vector[2, 0], joints_location_vector[2, 1]),
             (joints_location_vector[9, 0], joints_location_vector[9, 1]))

    ### interest points along the body:




    # head:
    head_top
    mid_head
    head_bottom

    # neck:
    mid_neck
    bottom_neck

    # shoulders:
    right_shoulder
    right_mid_way_shoulder
    left_shoulder
    left_mid_way_shoulder

    # hands:
    right_elbow
    left_elbow
    right_mid_shoulder_to_elbow
    left_mid_shoulder_to_elbow
    right_wrist
    left_wrist
    right_mid_elbow_to_wrist
    left_mid_elbow_to_wrist

    # pelvic:
    right_pelvic = (joints_location_vector[2, 0], joints_location_vector[2, 1])
    right_mid_way_pelvic
    left_pelvic = (joints_location_vector[3, 0], joints_location_vector[3, 1])
    left_mid_way_pelvic
    mid_pelvic

    # legs:
    right_knee = (joints_location_vector[1, 0], joints_location_vector[1, 1])
    left_knee = (joints_location_vector[4, 0], joints_location_vector[4, 1])
    right_mid_pelvic_to_knee
    left_mid_pelvic_to_knee
    right_ankle = (joints_location_vector[0, 0], joints_location_vector[0, 1])
    left_ankle = (joints_location_vector[5, 0], joints_location_vector[5, 1])
    right_mid_knee_to_ankle
    left_mid_knee_to_ankle

    # torso:
    right_shoulder_to_left_pelvic_1
    right_shoulder_to_left_pelvic_2
    right_shoulder_to_left_pelvic_3
    left_shoulder_to_right_pelvic_1
    left_shoulder_to_right_pelvic_2
    left_shoulder_to_right_pelvic_3
    mid_shoulders_to_mid_pelvic_1
    mid_shoulders_to_mid_pelvic_2
    mid_shoulders_to_mid_pelvic_3


    grabbed_human_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grabbed_human_image[:, :, 2] = 150
    grabbed_human_image[:, :, 1] = 150


    return top_mask, bottom_mask, head_cover_mask, leggings_mask, shoes_mask, overall_or_dress_mask


def human_body_cut(image, joints_location_vector):

    mask = np.zeros(image.shape[:2]).astype('uint8')

    human_skeleton_masks = human_body_probability_map(image, joints_location_vector)
    human_skeleton = human_skeleton_masks[0]  # first index of limb masks
    human_skeleton[human_skeleton>0] = 255
    cv2.imshow('l', human_skeleton)
    cv2.waitKey(0)

    #calculating Pscale - the scale in which the unsertainty masks are created:
    upper_left_corner, lower_right_corner = pose_bbox(joints_location_vector)
    size = np.array(lower_right_corner) - np.array(upper_left_corner)
    Pscale = max(size)*9 / (min(size)*2)

    # probably background mask:
    kernel = np.ones((3, 3), 'uint8')
    distance_mask = cv2.dilate(human_skeleton, kernel, iterations=Pscale*2)
    distance_mask = cv2.distanceTransform(distance_mask, cv2.DIST_L2, 3)
    cv2.normalize(distance_mask, distance_mask, 0, 1., cv2.NORM_MINMAX)
    mask[distance_mask > 0] = 3

    # probably foreground mask:
    kernel = np.ones((3, 3), 'uint8')
    distance_mask = cv2.dilate(human_skeleton, kernel, iterations=Pscale)
    inverse_distance_mask = cv2.distanceTransform(255-distance_mask, cv2.DIST_L2, 3)
    cv2.normalize(inverse_distance_mask, inverse_distance_mask, 0, 1., cv2.NORM_MINMAX)
    inverse_distance_mask = inverse_distance_mask ** 2
    # cv2.imshow('l', inverse_distance_mask)
    # cv2.waitKey(0)
    blurred = image.copy()
    blurred_image = image.copy()
    grad = ((Pscale**2) * 0.00001, (Pscale**2) * 0.0001, (Pscale**2) * 0.001)#, Pscale * 0.01)
    # print grad
    k = 3
    for mask_val in grad:
        blurred_image = cv2.GaussianBlur(blurred_image, (k, k), 0)
        blur_mask = np.zeros(inverse_distance_mask.shape)
        blur_mask[mask_val<=inverse_distance_mask] = 1
        blurred[blur_mask>0] = blurred_image[blur_mask>0]
        k += 2
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    mask[distance_mask > 0] = 2

    # foreground mask:
    mask[human_skeleton > 0] = 1

    # cv2.imshow('l', mask * 40)
    # cv2.waitKey(0)

    rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = bgdmodel # np.zeros((1, 65), np.float64)
    itercount = 1
    returned_mask, bgdModel, fgdModel = cv2.grabCut(blurred, mask.astype('uint8'), rect, bgdmodel, fgdmodel, itercount, cv2.GC_INIT_WITH_MASK)
    mask = np.where((returned_mask == 2) | (returned_mask == 0), 0, 1).astype('uint8')
    grabbed_human_image = image * mask[:, :, np.newaxis]

    # image0 = plot_image_skeleton_for_testing(image.copy(), joints_location_vector)
    # cv2.imshow('l', np.hstack([image0, image]))
    # cv2.waitKey(0)
    return mask, grabbed_human_image


def remap(x, oMin, oMax, nMin, nMax):
    '''
    :param x:
    :param oMin:
    :param oMax:
    :param nMin:
    :param nMax:
    :return:
    '''
    #range check
    if oMin == oMax:
        print "Warning: Zero input range"
        return None

    if nMin == nMax:
        print "Warning: Zero output range"
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min(oMin, oMax)
    oldMax = max(oMin, oMax)
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False
    newMin = min(nMin, nMax)
    newMax = max(nMin, nMax)
    if not newMin == nMin :
        reverseOutput = True

# new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    portion = (x-oldMin)*(float(newMax-newMin)/(oldMax-oldMin))
    if reverseInput:
        portion = (oldMax-x)*(float(newMax-newMin)/(oldMax-oldMin))

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion
    result = np.array(result).astype('uint8')
    return result


def whiten_image(image):
    '''
    :param image: uint8 image grayscale or BGR
    :return:
    '''
    whitend_image = []
    # if grayscale:
    if len(image.shape) == 1:
        oMax = image.max()
        nMax = 255
        dScale = nMax - oMax
        if dScale > 0:
            whitend_image = np.zeros(image.shape)
            oMin = image.min()
            nMin = oMin
            whitend_image = remap(image, oMin, oMax, nMin, nMax)
        else:
            whitend_image = image

    # if BGR:
    elif len(image.shape) == 3:
        oMax = image.max()
        nMax = 255
        dScale = nMax - oMax
        if dScale > 0:
            whitend_image = np.zeros(image.shape)
            for i in range(3):
                oMin = image[:, :, i].min()
                nMin = oMin
                oMax = image[:, :, i].max()
                nMax = oMax + dScale
                whitend_image[:, :, i] = remap(image[:, :, i], oMin, oMax, nMin, nMax)
        else:
            whitend_image = image

    else:
        print 'Error: input is not a 3 channle image nore a grayscale (1 ch) image!'
    return whitend_image.astype('uint8')


def PIL2array(image):
    '''
    :param image:
    :return:
    '''
    return np.array(image.getdata(), 'uint8').reshape(image.size[1], image.size[0], 3)


def array2PIL(arr, size):
    '''
    :param arr:
    :param size:
    :return:
    '''
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), 'uint8')]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def determine_input_image_size_by_primal_dataset(path_to_primal_dataset):
    '''
    :param path_to_primal_dataset:
    :return:
    '''
    only_files = [f for f in os.listdir(path_to_primal_dataset) if os.path.isfile(os.path.join(path_to_primal_dataset, f))]
    h = 0
    w = 0
    for file_name in only_files:
        h_new, w_new = cv2.imread(path_to_primal_dataset + file_name).shape[:2]
        if h_new > h:
            h = h_new
            print 'changed h'
        if w_new > w:
            w = w_new
            print 'changed w'
    A = max(h, w)
    input_image_size = (A, A)
    return input_image_size


def vflipped_image_and_vectors(image, joints_location_vector):
    '''
    :param image:
    :param joints_location_vector:
    :return:
    '''
    joints_location_vector = joints_location_vector.astype(int)
    im = image.copy()
    # plot_image_skeleton_for_testing(image, joints_location_vector)
    shape = image.shape[:2]
    flipped_image = np.fliplr(im).astype('uint8')
    flipped_joints_location_vector = joints_location_vector.copy()
    flipped_joints_location_vector[:, 0] = -flipped_joints_location_vector[:, 0] + shape[1]

    # TODO:switch right and lef extremities X-coordinate: DONE!
    Right_ankle = flipped_joints_location_vector[0, 0]
    Right_knee = flipped_joints_location_vector[1, 0]
    Right_hip = flipped_joints_location_vector[2, 0]
    Left_hip = flipped_joints_location_vector[3, 0]
    Left_knee = flipped_joints_location_vector[4, 0]
    Left_ankle = flipped_joints_location_vector[5, 0]
    Right_wrist = flipped_joints_location_vector[6, 0]
    Right_elbow = flipped_joints_location_vector[7, 0]
    Right_shoulder = flipped_joints_location_vector[8, 0]
    Left_shoulder = flipped_joints_location_vector[9, 0]
    Left_elbow = flipped_joints_location_vector[10, 0]
    Left_wrist = flipped_joints_location_vector[11, 0]
    # Neck
    # Head top
    flipped_joints_location_vector[0, 0] = Left_ankle
    flipped_joints_location_vector[1, 0] = Left_knee
    flipped_joints_location_vector[2, 0] = Left_hip
    flipped_joints_location_vector[3, 0] = Right_hip
    flipped_joints_location_vector[4, 0] = Right_knee
    flipped_joints_location_vector[5, 0] = Right_ankle
    flipped_joints_location_vector[6, 0] = Left_wrist
    flipped_joints_location_vector[7, 0] = Left_elbow
    flipped_joints_location_vector[8, 0] = Left_shoulder
    flipped_joints_location_vector[9, 0] = Right_shoulder
    flipped_joints_location_vector[10, 0] = Right_elbow
    flipped_joints_location_vector[11, 0] = Right_wrist

    # TODO:switch right and lef extremities Y-coordinate: DONE!
    Right_ankle = flipped_joints_location_vector[0, 1]
    Right_knee = flipped_joints_location_vector[1, 1]
    Right_hip = flipped_joints_location_vector[2, 1]
    Left_hip = flipped_joints_location_vector[3, 1]
    Left_knee = flipped_joints_location_vector[4, 1]
    Left_ankle = flipped_joints_location_vector[5, 1]
    Right_wrist = flipped_joints_location_vector[6, 1]
    Right_elbow = flipped_joints_location_vector[7, 1]
    Right_shoulder = flipped_joints_location_vector[8, 1]
    Left_shoulder = flipped_joints_location_vector[9, 1]
    Left_elbow = flipped_joints_location_vector[10, 1]
    Left_wrist = flipped_joints_location_vector[11, 1]
    # Neck
    # Head top
    flipped_joints_location_vector[0, 1] = Left_ankle
    flipped_joints_location_vector[1, 1] = Left_knee
    flipped_joints_location_vector[2, 1] = Left_hip
    flipped_joints_location_vector[3, 1] = Right_hip
    flipped_joints_location_vector[4, 1] = Right_knee
    flipped_joints_location_vector[5, 1] = Right_ankle
    flipped_joints_location_vector[6, 1] = Left_wrist
    flipped_joints_location_vector[7, 1] = Left_elbow
    flipped_joints_location_vector[8, 1] = Left_shoulder
    flipped_joints_location_vector[9, 1] = Right_shoulder
    flipped_joints_location_vector[10, 1] = Right_elbow
    flipped_joints_location_vector[11, 1] = Right_wrist

    # TODO:switch right and lef extremities is-visible: DONE!
    Right_ankle = flipped_joints_location_vector[0, 2]
    Right_knee = flipped_joints_location_vector[1, 2]
    Right_hip = flipped_joints_location_vector[2, 2]
    Left_hip = flipped_joints_location_vector[3, 2]
    Left_knee = flipped_joints_location_vector[4, 2]
    Left_ankle = flipped_joints_location_vector[5, 2]
    Right_wrist = flipped_joints_location_vector[6, 2]
    Right_elbow = flipped_joints_location_vector[7, 2]
    Right_shoulder = flipped_joints_location_vector[8, 2]
    Left_shoulder = flipped_joints_location_vector[9, 2]
    Left_elbow = flipped_joints_location_vector[10, 2]
    Left_wrist = flipped_joints_location_vector[11, 2]
    # Neck
    # Head top
    flipped_joints_location_vector[0, 2] = Left_ankle
    flipped_joints_location_vector[1, 2] = Left_knee
    flipped_joints_location_vector[2, 2] = Left_hip
    flipped_joints_location_vector[3, 2] = Right_hip
    flipped_joints_location_vector[4, 2] = Right_knee
    flipped_joints_location_vector[5, 2] = Right_ankle
    flipped_joints_location_vector[6, 2] = Left_wrist
    flipped_joints_location_vector[7, 2] = Left_elbow
    flipped_joints_location_vector[8, 2] = Left_shoulder
    flipped_joints_location_vector[9, 2] = Right_shoulder
    flipped_joints_location_vector[10, 2] = Right_elbow
    flipped_joints_location_vector[11, 2] = Right_wrist

    # plot_image_skeleton_for_testing(flipped_image, flipped_joints_location_vector)
    return flipped_image, flipped_joints_location_vector


def pose_bbox(joints_location_vector):
    '''
    :param joints_location_vector:
    :return:
    '''
    upper_left_corner = [int(joints_location_vector[:, 0].min()), int(joints_location_vector[:, 1].min())]
    lower_right_corner = [int(joints_location_vector[:, 0].max()), int(joints_location_vector[:, 1].max())]
    for i in range(2):
        if upper_left_corner[i] < 0:
            upper_left_corner[i] = 0
        if lower_right_corner[i] < 0:
            lower_right_corner[i] = 0
    return upper_left_corner, lower_right_corner


def crop_enlarged_bbox(image, joints_location_vector):
    '''
    :param image:
    :param joints_location_vector:
    :return:
    '''
    # joints_location_vector = abs(joints_location_vector)
    shape = image.shape[:2]
    upper_left_corner, lower_right_corner = pose_bbox(joints_location_vector)
    margine = np.abs(np.array(upper_left_corner, dtype='float16') - np.array(lower_right_corner, dtype='float16'))
    margine = 0.1 * np.min(margine) / np.max(margine)
    #cropping the image:
    delta_margine_1 = int(margine*(upper_left_corner[0] + (shape[0] - lower_right_corner[0]))/2)
    delta_margine_2 = int(margine*(upper_left_corner[1] + (shape[1] - lower_right_corner[1]))/2)
    if delta_margine_1 < (lower_right_corner[0] - upper_left_corner[0]) / 8:
        delta_margine_1 = 0
    if delta_margine_2 < (lower_right_corner[1] - upper_left_corner[1]) / 8:
        delta_margine_2 = 0

    horizontal_1 = delta_margine_1 #upper_left_corner[1] - delta_margine_1
    if horizontal_1 < 0:
        horizontal_1 = 0
    horizontal_2 = shape[0] - delta_margine_1 #lower_right_corner[1] + delta_margine_1
    if horizontal_2 < 0:
        horizontal_2 = 0
    vertical_1 = delta_margine_2 #upper_left_corner[0] - delta_margine_2
    if vertical_1 < 0:
        vertical_1 = 0
    vertical_2 = shape[1] - delta_margine_2 #lower_right_corner[0] + delta_margine_2
    if vertical_2 < 0:
        vertical_2 = 0

    tight_image = image[horizontal_1:horizontal_2, vertical_1:vertical_2, :]

    # moving the joints coordinate:
    tight_joints_location_vector = joints_location_vector.copy()
    tight_joints_location_vector[:, 0] = tight_joints_location_vector[:, 0] - vertical_1 #- delta_margine_2
    tight_joints_location_vector[:, 1] = tight_joints_location_vector[:, 1] - horizontal_1#- delta_margine_1

    # plot_image_skeleton_for_testing(tight_image, tight_joints_location_vector)

    return tight_image, tight_joints_location_vector


def add_brim_to_image_and_move_joints_vector(image, joints_location_vector, max_shape):
    '''
    :param image:
    :param output_size:
    :param joints_location_vector:
    :return:
    '''

    shifting = True
    output_images_size = image.shape[:2]
    #shrinking oversized images:
    if image.shape[0] > max_shape[0] or image.shape[1] > max_shape[1]:
        image, joints_location_vector = crop_enlarged_bbox(image.copy(), joints_location_vector.copy())
        if image.shape[0] > image.shape[1]:
            # output_images_size = (image.shape[1]*max_shape[0]/image.shape[0], max_shape[0])
            scale = 1.0*max_shape[0]/image.shape[0]
        else:
            # output_images_size = (max_shape[1], image.shape[0]*max_shape[1]/image.shape[1])
            scale = 1.0*max_shape[1]/image.shape[1]

        joints_location_vector[:, 0] = joints_location_vector[:, 0] * scale
        joints_location_vector[:, 1] = joints_location_vector[:, 1] * scale
        image = cv2.resize(image.copy(), (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        output_images_size = image.shape[:2]

    border_type = cv2.BORDER_REPLICATE
    # getting image size:
    h_max, w_max = max_shape #output_size
    h, w = output_images_size
    w_shift = w_max - w
    h_shift = h_max - h

    if h_shift > h_max / 3:
        cornerize1 = True
    else:
        cornerize1 = False
    if w_shift > w_max / 3:
        cornerize2 = True
    else:
        cornerize2 = False
    # print w_shift
    # print h_shift

    # rotation matrixes:
    R270 = np.array([[0, -1], [1, 0]])
    R180 = np.array([[-1, 0], [0, -1]])
    R90 = np.array([[0, 1], [-1, 0]])
    output_images_and_vectors = []

    # 0: central placement
    ##########################################################
    ##########################################################
    ##########################################################
    # no rotate:
    location_vector = joints_location_vector.copy()
    output_image = cv2.copyMakeBorder(image, h_shift/2, h_shift/2, w_shift/2, w_shift/2, border_type)
    location_vector[:, 0] = joints_location_vector[:, 0] + w_shift/2
    location_vector[:, 1] = joints_location_vector[:, 1] + h_shift/2
    joints_location_vector_0 = location_vector.astype(int)
    output_image_0 = output_image.copy()
    output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
    # print joints_location_vector_0
    ####################################################################
    # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
    ###################################################################

    # # 90.deg CCW rotation:
    # joints_location_vector_90 = joints_location_vector_0.copy()
    # for i in range(14):
    #     vector_90 = np.dot(R90, np.array([joints_location_vector_0[i, 0] - h_max/2,
    #                                       joints_location_vector_0[i, 1] - w_max/2]))
    #     joints_location_vector_90[i, 0] = vector_90[0] + h_max/2
    #     joints_location_vector_90[i, 1] = vector_90[1] + w_max/2
    # output_image_90 = ndimage.rotate(output_image.copy(), 90)
    # output_images_and_vectors.append([cv2.resize(output_image_90, max_shape), joints_location_vector_90])
    # # print joints_location_vector_90
    # ####################################################################
    # # plot_image_skeleton_for_testing(output_image_90, joints_location_vector_90)
    # ###################################################################
    #
    # # 180.deg CCW rotation:
    # joints_location_vector_180 = joints_location_vector_0.copy()
    # for i in range(14):
    #     vector_180 = np.dot(R180, np.array([joints_location_vector_0[i, 0] - w_max/2,
    #                                         joints_location_vector_0[i, 1] - h_max/2]))
    #     joints_location_vector_180[i, 0] = vector_180[0] + w_max/2
    #     joints_location_vector_180[i, 1] = vector_180[1] + h_max/2
    # output_image_180 = ndimage.rotate(output_image.copy(), 180)
    # output_images_and_vectors.append([cv2.resize(output_image_180, max_shape), joints_location_vector_180])
    # # print joints_location_vector_180
    # ####################################################################
    # # plot_image_skeleton_for_testing(output_image_180, joints_location_vector_180)
    # ###################################################################
    #
    # # 270.deg CCW rotation:
    # joints_location_vector_270 = joints_location_vector_0.copy()
    # for i in range(14):
    #     vector_270 = np.dot(R270, np.array([joints_location_vector_0[i, 0] - w_max/2,
    #                                         joints_location_vector_0[i, 1] - h_max/2]))
    #     joints_location_vector_270[i, 0] = vector_270[0] + w_max/2
    #     joints_location_vector_270[i, 1] = vector_270[1] + h_max/2
    # output_image_270 = ndimage.rotate(output_image.copy(), 270)
    # output_images_and_vectors.append([cv2.resize(output_image_270, max_shape), joints_location_vector_270])
    # # print joints_location_vector_270
    # ####################################################################
    # # plot_image_skeleton_for_testing(output_image_270, joints_location_vector_270)
    # ###################################################################
    #
    # ##########################################################
    # ##########################################################
    # ##########################################################


    if shifting:
        if cornerize1:
            # 1: upper left corner placement
            ##########################################################
            ##########################################################
            ##########################################################
            # no rotate:
            location_vector = joints_location_vector.copy()
            output_image = cv2.copyMakeBorder(image, 0, h_shift, 0, w_shift, border_type)
            location_vector[:, 0] = joints_location_vector[:, 0]
            location_vector[:, 1] = joints_location_vector[:, 1]
            joints_location_vector_0 = location_vector.astype(int)
            output_image_0 = output_image.copy()
            output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
            # print joints_location_vector_0
            ####################################################################
            # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
            ###################################################################

            # # 90.deg CCW rotation:
            # joints_location_vector_90 = joints_location_vector_0.copy()
            # for i in range(14):
            #     vector_90 = np.dot(R90, np.array([joints_location_vector_0[i, 0] - w_max/2,
            #                                       joints_location_vector_0[i, 1] - h_max/2]))
            #     joints_location_vector_90[i, 0] = vector_90[0] + w_max/2
            #     joints_location_vector_90[i, 1] = vector_90[1] + h_max/2
            # output_image_90 = ndimage.rotate(output_image.copy(), 90)
            # output_images_and_vectors.append([cv2.resize(output_image_90, max_shape), joints_location_vector_90])
            # # print joints_location_vector_90
            # ####################################################################
            # # plot_image_skeleton_for_testing(output_image_90, joints_location_vector_90)
            # ###################################################################
            #
            # # 180.deg CCW rotation:
            # joints_location_vector_180 = joints_location_vector_0.copy()
            # for i in range(14):
            #     vector_180 = np.dot(R180, np.array([joints_location_vector_0[i, 0] - w_max/2,
            #                                         joints_location_vector_0[i, 1] - h_max/2]))
            #     joints_location_vector_180[i, 0] = vector_180[0] + w_max/2
            #     joints_location_vector_180[i, 1] = vector_180[1] + h_max/2
            # output_image_180 = ndimage.rotate(output_image.copy(), 180)
            # output_images_and_vectors.append([cv2.resize(output_image_180, max_shape), joints_location_vector_180])
            # # print joints_location_vector_180
            # ####################################################################
            # # plot_image_skeleton_for_testing(output_image_180, joints_location_vector_180)
            # ###################################################################
            #
            # # 270.deg CCW rotation:
            # joints_location_vector_270 = joints_location_vector_0.copy()
            # for i in range(14):
            #     vector_270 = np.dot(R270, np.array([joints_location_vector_0[i, 0] - w_max/2,
            #                                         joints_location_vector_0[i, 1] - h_max/2]))
            #     joints_location_vector_270[i, 0] = vector_270[0] + w_max/2
            #     joints_location_vector_270[i, 1] = vector_270[1] + h_max/2
            # output_image_270 = ndimage.rotate(output_image.copy(), 270)
            # output_images_and_vectors.append([cv2.resize(output_image_270, max_shape), joints_location_vector_270])
            # # print joints_location_vector_270
            # ####################################################################
            # # plot_image_skeleton_for_testing(output_image_270, joints_location_vector_270)
            # ###################################################################

            ##########################################################
            ##########################################################
            ##########################################################

            if w_shift > w_max / 3:
                # 2: lower left corner placement
                ##########################################################
                ##########################################################
                ##########################################################
                # no rotate:
                location_vector = joints_location_vector.copy()
                output_image = cv2.copyMakeBorder(image, h_shift, 0, 0, w_shift, border_type)
                location_vector[:, 0] = joints_location_vector[:, 0]
                location_vector[:, 1] = joints_location_vector[:, 1] + h_shift
                joints_location_vector_0 = location_vector.astype(int)
                output_image_0 = output_image.copy()
                output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
                # print joints_location_vector_0
                ####################################################################
                # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
                ###################################################################

                # # 90.deg CCW rotation:
                # joints_location_vector_90 = joints_location_vector_0.copy()
                # for i in range(14):
                #     vector_90 = np.dot(R90, np.array([joints_location_vector_0[i, 0] - w_max/2,
                #                                       joints_location_vector_0[i, 1] - h_max/2]))
                #     joints_location_vector_90[i, 0] = vector_90[0] + w_max/2
                #     joints_location_vector_90[i, 1] = vector_90[1] + h_max/2
                # output_image_90 = ndimage.rotate(output_image.copy(), 90)
                # output_images_and_vectors.append([cv2.resize(output_image_90, max_shape), joints_location_vector_90])
                # # print joints_location_vector_90
                # ####################################################################
                # # plot_image_skeleton_for_testing(output_image_90, joints_location_vector_90)
                # ###################################################################
                #
                # # 180.deg CCW rotation:
                # joints_location_vector_180 = joints_location_vector_0.copy()
                # for i in range(14):
                #     vector_180 = np.dot(R180, np.array([joints_location_vector_0[i, 0] - w_max/2,
                #                                       joints_location_vector_0[i, 1] - h_max/2]))
                #     joints_location_vector_180[i, 0] = vector_180[0] + w_max/2
                #     joints_location_vector_180[i, 1] = vector_180[1] + h_max/2
                # output_image_180 = ndimage.rotate(output_image.copy(), 180)
                # output_images_and_vectors.append([cv2.resize(output_image_180, max_shape), joints_location_vector_180])
                # # print joints_location_vector_180
                # ####################################################################
                # # plot_image_skeleton_for_testing(output_image_180, joints_location_vector_180)
                # ###################################################################
                #
                # # 270.deg CCW rotation:
                # joints_location_vector_270 = joints_location_vector_0.copy()
                # for i in range(14):
                #     vector_270 = np.dot(R270, np.array([joints_location_vector_0[i, 0] - w_max/2,
                #                                       joints_location_vector_0[i, 1] - h_max/2]))
                #     joints_location_vector_270[i, 0] = vector_270[0] + w_max/2
                #     joints_location_vector_270[i, 1] = vector_270[1] + h_max/2
                # output_image_270 = ndimage.rotate(output_image.copy(), 270)
                # output_images_and_vectors.append([cv2.resize(output_image_270, max_shape), joints_location_vector_270])
                # # print joints_location_vector_270
                # ####################################################################
                # # plot_image_skeleton_for_testing(output_image_270, joints_location_vector_270)
                # ###################################################################

        ##########################################################
        ##########################################################
        ##########################################################

        if cornerize2:
            # 3: upper right corner placement
            ##########################################################
            ##########################################################
            ##########################################################
            # no rotate:
            location_vector = joints_location_vector.copy()
            output_image = cv2.copyMakeBorder(image, 0, h_shift, w_shift, 0, border_type)
            location_vector[:, 0] = joints_location_vector[:, 0] + w_shift
            location_vector[:, 1] = joints_location_vector[:, 1]
            joints_location_vector_0 = location_vector.astype(int)
            output_image_0 = output_image.copy()
            output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
            # print joints_location_vector_0
            ####################################################################
            # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
            ###################################################################

            # # 90.deg CCW rotation:
            # joints_location_vector_90 = joints_location_vector_0.copy()
            # for i in range(14):
            #     vector_90 = np.dot(R90, np.array([joints_location_vector_0[i, 0] - w_max/2,
            #                                       joints_location_vector_0[i, 1] - h_max/2]))
            #     joints_location_vector_90[i, 0] = vector_90[0] + w_max/2
            #     joints_location_vector_90[i, 1] = vector_90[1] + h_max/2
            # output_image_90 = ndimage.rotate(output_image.copy(), 90)
            # output_images_and_vectors.append([cv2.resize(output_image_90, max_shape), joints_location_vector_90])
            # # print joints_location_vector_90
            # ####################################################################
            # # plot_image_skeleton_for_testing(output_image_90, joints_location_vector_90)
            # ###################################################################
            #
            # # 180.deg CCW rotation:
            # joints_location_vector_180 = joints_location_vector_0.copy()
            # for i in range(14):
            #     vector_180 = np.dot(R180, np.array([joints_location_vector_0[i, 0] - w_max/2,
            #                                       joints_location_vector_0[i, 1] - h_max/2]))
            #     joints_location_vector_180[i, 0] = vector_180[0] + w_max/2
            #     joints_location_vector_180[i, 1] = vector_180[1] + h_max/2
            # output_image_180 = ndimage.rotate(output_image.copy(), 180)
            # output_images_and_vectors.append([cv2.resize(output_image_180, max_shape), joints_location_vector_180])
            # # print joints_location_vector_180
            # ####################################################################
            # # plot_image_skeleton_for_testing(output_image_180, joints_location_vector_180)
            # ###################################################################
            #
            # # 270.deg CCW rotation:
            # joints_location_vector_270 = joints_location_vector_0.copy()
            # for i in range(14):
            #     vector_270 = np.dot(R270, np.array([joints_location_vector_0[i, 0] - w_max/2,
            #                                       joints_location_vector_0[i, 1] - h_max/2]))
            #     joints_location_vector_270[i, 0] = vector_270[0] + w_max/2
            #     joints_location_vector_270[i, 1] = vector_270[1] + h_max/2
            # output_image_270 = ndimage.rotate(output_image.copy(), 270)
            # output_images_and_vectors.append([cv2.resize(output_image_270, max_shape), joints_location_vector_270])
            # # print joints_location_vector_270
            # ####################################################################
            # # plot_image_skeleton_for_testing(output_image_270, joints_location_vector_270)
            # ###################################################################

            ##########################################################
            ##########################################################
            ##########################################################

            # 4: lower right corner placement
            if h_shift > h_max / 3:
                ##########################################################
                ##########################################################
                ##########################################################
                # no rotate:
                location_vector = joints_location_vector.copy()
                output_image = cv2.copyMakeBorder(image, h_shift, 0, w_shift, 0, border_type)
                location_vector[:, 0] = joints_location_vector[:, 0] + w_shift
                location_vector[:, 1] = joints_location_vector[:, 1] + h_shift
                joints_location_vector_0 = location_vector.astype(int)
                output_image_0 = output_image.copy()
                output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
                # print joints_location_vector_0
                ####################################################################
                # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
                ###################################################################

                # # 90.deg CCW rotation:
                # joints_location_vector_90 = joints_location_vector_0.copy()
                # for i in range(14):
                #     vector_90 = np.dot(R90, np.array([joints_location_vector_0[i, 0] - w_max/2,
                #                                       joints_location_vector_0[i, 1] - h_max/2]))
                #     joints_location_vector_90[i, 0] = vector_90[0] + w_max/2
                #     joints_location_vector_90[i, 1] = vector_90[1] + h_max/2
                # output_image_90 = ndimage.rotate(output_image.copy(), 90)
                # output_images_and_vectors.append([cv2.resize(output_image_90, max_shape), joints_location_vector_90])
                # # print joints_location_vector_90
                # ####################################################################
                # # plot_image_skeleton_for_testing(output_image_90, joints_location_vector_90)
                # ###################################################################
                #
                # # 180.deg CCW rotation:
                # joints_location_vector_180 = joints_location_vector_0.copy()
                # for i in range(14):
                #     vector_180 = np.dot(R180, np.array([joints_location_vector_0[i, 0] - w_max/2,
                #                                       joints_location_vector_0[i, 1] - h_max/2]))
                #     joints_location_vector_180[i, 0] = vector_180[0] + w_max/2
                #     joints_location_vector_180[i, 1] = vector_180[1] + h_max/2
                # output_image_180 = ndimage.rotate(output_image.copy(), 180)
                # output_images_and_vectors.append([cv2.resize(output_image_180, max_shape), joints_location_vector_180])
                # # print joints_location_vector_180
                # ####################################################################
                # # plot_image_skeleton_for_testing(output_image_180, joints_location_vector_180)
                # ###################################################################
                #
                # # 270.deg CCW rotation:
                # joints_location_vector_270 = joints_location_vector_0.copy()
                # for i in range(14):
                #     vector_270 = np.dot(R270, np.array([joints_location_vector_0[i, 0] - w_max/2,
                #                                       joints_location_vector_0[i, 1] - h_max/2]))
                #     joints_location_vector_270[i, 0] = vector_270[0] + w_max/2
                #     joints_location_vector_270[i, 1] = vector_270[1] + h_max/2
                # output_image_270 = ndimage.rotate(output_image.copy(), 270)
                # output_images_and_vectors.append([cv2.resize(output_image_270, max_shape), joints_location_vector_270])
                # # print joints_location_vector_270
                # ####################################################################
                # # plot_image_skeleton_for_testing(output_image_270, joints_location_vector_270)
                # ###################################################################

                ##########################################################
                ##########################################################
                ##########################################################


    # sizing up small images:
    if (cornerize1 or cornerize2) and min(1.0 * w_shift / w_max, 1.0 * h_shift / h_max) > 1./3:
        if w_shift > h_shift:
            output_images_size = (w*h_max/h, h_max)
            scale = 1.0 * h_max/h
            w_shift = w_max - w*h_max/h
            h_shift = 0
        else:
            output_images_size = (w_max, h*w_max/w)
            scale = 1.0 * w_max/w
            w_shift = 0
            h_shift = h_max - h*w_max/w

        joints_location_vector_d = joints_location_vector.copy()
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        joints_location_vector_d[:, 0] = joints_location_vector[:, 0] * scale
        joints_location_vector_d[:, 1] = joints_location_vector[:, 1] * scale

        # 0: central placement
        ##########################################################
        ##########################################################
        ##########################################################
        # no rotate:
        location_vector = joints_location_vector_d.copy()
        output_image = cv2.copyMakeBorder(image, h_shift/2, h_shift/2, w_shift/2, w_shift/2, border_type)
        location_vector[:, 0] = joints_location_vector_d[:, 0] + w_shift/2
        location_vector[:, 1] = joints_location_vector_d[:, 1] + h_shift/2
        joints_location_vector_0 = location_vector.astype(int)
        output_image_0 = output_image.copy()
        output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
        # print joints_location_vector_0
        ####################################################################
        # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
        ###################################################################

        # # 90.deg CCW rotation:
        # joints_location_vector_90 = joints_location_vector_0.copy()
        # for i in range(14):
        #     vector_90 = np.dot(R90, np.array([joints_location_vector_0[i, 0] - w_max/2,
        #                                       joints_location_vector_0[i, 1] - h_max/2]))
        #     joints_location_vector_90[i, 0] = vector_90[0] + w_max/2
        #     joints_location_vector_90[i, 1] = vector_90[1] + h_max/2
        # output_image_90 = ndimage.rotate(output_image.copy(), 90)
        # output_images_and_vectors.append([cv2.resize(output_image_90, max_shape), joints_location_vector_90])
        # # print joints_location_vector_90
        # ####################################################################
        # # plot_image_skeleton_for_testing(output_image_90, joints_location_vector_90)
        # ###################################################################
        #
        # # 180.deg CCW rotation:
        # joints_location_vector_180 = joints_location_vector_0.copy()
        # for i in range(14):
        #     vector_180 = np.dot(R180, np.array([joints_location_vector_0[i, 0] - w_max/2,
        #                                         joints_location_vector_0[i, 1] - h_max/2]))
        #     joints_location_vector_180[i, 0] = vector_180[0] + w_max/2
        #     joints_location_vector_180[i, 1] = vector_180[1] + h_max/2
        # output_image_180 = ndimage.rotate(output_image.copy(), 180)
        # output_images_and_vectors.append([cv2.resize(output_image_180, max_shape), joints_location_vector_180])
        # # print joints_location_vector_180
        # ####################################################################
        # # plot_image_skeleton_for_testing(output_image_180, joints_location_vector_180)
        # ###################################################################
        #
        # # 270.deg CCW rotation:
        # joints_location_vector_270 = joints_location_vector_0.copy()
        # for i in range(14):
        #     vector_270 = np.dot(R270, np.array([joints_location_vector_0[i, 0] - w_max/2,
        #                                         joints_location_vector_0[i, 1] - h_max/2]))
        #     joints_location_vector_270[i, 0] = vector_270[0] + w_max/2
        #     joints_location_vector_270[i, 1] = vector_270[1] + h_max/2
        # output_image_270 = ndimage.rotate(output_image.copy(), 270)
        # output_images_and_vectors.append([cv2.resize(output_image_270, max_shape), joints_location_vector_270])
        # # print joints_location_vector_270
        # ####################################################################
        # # plot_image_skeleton_for_testing(output_image_270, joints_location_vector_270)
        # ###################################################################

        ##########################################################
        ##########################################################
        ##########################################################


    return output_images_and_vectors



def coloring_images_and_vectors(image):
    '''
    :param image:
    :param joints_location_vector:
    :return:
    '''

    enhancement_factors = [0.4, 1.4]
    output_images = []
    # image = Image.fromarray(whiten_image(image))
    # cv2.imshow('l', image)
    # cv2.waitKey(0)
    # image = cv2.equalizeHist(image)
    # cv2.imshow('l', image)
    # cv2.waitKey(0)
    output_images.append(image)
    enhancers = []
    enhancers.append(ImageEnhance.Sharpness(Image.fromarray(image)))
    enhancers.append(ImageEnhance.Brightness(Image.fromarray(image)))
    enhancers.append(ImageEnhance.Contrast(Image.fromarray(image)))
    enhancers.append(ImageEnhance.Color(Image.fromarray(image)))
    for enhancer in enhancers:
        for factor in enhancement_factors:
            im = PIL2array(enhancer.enhance(factor))
            output_images.append(im)
            # cv2.imshow('l', PIL2array(enhancer.enhance(factor)))
            # cv2.waitKey(0)
    return output_images


def color_delta(image, delta=30):

    images = [image]
    csv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    drange = []

    for i in range(3):
        r = (np.amax(csv_image[:, :, i]) - delta) - (np.amin(csv_image[:, :, i]) + delta)
        steps = r/delta
        if steps%2 > 0:
            steps = steps - 1
        if steps > 0:
            drange = range(-delta*steps/2, delta*steps/2+1, delta)
            for d in drange:
                if (np.amin(csv_image[:, :, i] + d) > 0 and np.amax(csv_image[:, :, i] + d) < 255) and d != 0:
                    dcsv_image = csv_image.copy()
                    dcsv_image[:, :, i] = abs(csv_image[:, :, i] + d)
                    # cv2.imshow('p', np.hstack([image, cv2.cvtColor(dcsv_image, cv2.COLOR_HSV2BGR)]))
                    # cv2.waitKey(0)
                    images.append(cv2.cvtColor(dcsv_image, cv2.COLOR_HSV2BGR))

    return images


def variation_list_for_one_image(image, joints_location_vector, sample_image_size):
    '''
    :param image:
    :param output_size:
    :param joints_location_vector:
    :return:
    '''

    H = []
    # V = coloring_images_and_vectors(image)
    V = color_delta(image)
    for v in V:
        im = v.copy()
        # cv2.imshow('L', im)
        # cv2.waitKey(0)
        H1 = add_brim_to_image_and_move_joints_vector(im.copy(), joints_location_vector.copy(), sample_image_size)
        # plot_image_skeleton_for_testing(im.copy(), joints_location_vector.copy())
        fim, fvec = vflipped_image_and_vectors(im.copy(), joints_location_vector.copy())
        H2 = add_brim_to_image_and_move_joints_vector(fim.copy(), fvec.copy(), sample_image_size)
        # plot_image_skeleton_for_testing(fim.copy(), fvec.copy())
        H = H + H1 + H2
        # for i in range(len(H)):
        #     plot_image_skeleton_for_testing(H[i][0], H[i][1])
    output_images_and_vectors = H
    return output_images_and_vectors


def import_and_merge_datasets():
    '''
    :return:
    '''

    # datasets upload:
    sample_image_size = (192, 192)
    # lsp (sport pose dataset - 2K samples):
    vecs1 = sio.loadmat('lsp_dataset/joints.mat')['joints']
    vecs1 = np.transpose(vecs1, (2, 1, 0)).astype('int')
    # lspet (extreme sport pose dataset - 10K samples):
    vecs2 = sio.loadmat('lspet_dataset/joints.mat')['joints']
    vecs2 = np.transpose(vecs2, (2, 0, 1)).astype('int')
    # fashionista (paperdoll pose dataset - 685 samples):
    vecs3 = h5py.File('fashionista_dataset/fashionista_joints.hdf5')['joints']
    vecs3 = np.transpose(vecs3, (2, 0, 1)).astype('int')

    print vecs1.shape, vecs2.shape, vecs3.shape

    # output list of joints file:
    joints_list_filename = 'TG_pose_joints'
    # creating a directory for the dataset images (augmented) and their pose (joints points) vectors:
    dataset_directory_name = 'TG_pose_dataset'
    current_directory_name = os.getcwd()
    directory_path = current_directory_name + '/' + dataset_directory_name
    if not os.path.exists(directory_path):
        os.mkdir(dataset_directory_name)

    # matching images to vectors, cropping, resizing and listing up:
    vecs_dataset = []
    index = 0

    counter1 = 0
    # fashionista dataset:
    for v3 in vecs3:
        v3 = np.array(v3)
        counter1 += 1
        im = cv2.imread('fashionista_dataset/images/im' + str(counter1).zfill(3) + '.jpg')
        if (im is not None) and \
                (v3[:, 1].max() <= im.shape[0] and v3[:, 0].max() <= im.shape[1]) and \
                (v3[:, 1].min() >= 0 and v3[:, 0].min() >= 0):
            images_and_vectors_list = variation_list_for_one_image(im, v3, sample_image_size)
            for varietion in images_and_vectors_list:
                vecs_dataset.append(varietion[1])
                cv2.imwrite(directory_path + '/TG_pose_im_' + str(index) + '.jpg', varietion[0])
                index += 1
        print counter1

    # counter2 = 0
    # # lsp dataset:
    # for v1 in vecs1:
    #     v1 = np.array(v1)
    #     counter2 += 1
    #     im = cv2.imread('lsp_dataset/images/im' + str(counter2).zfill(4) + '.jpg')
    #     if (im is not None) and \
    #             (v1[:, 1].max() <= im.shape[0] and v1[:, 0].max() <= im.shape[1]) and \
    #             (v1[:, 1].min() >= 0 and v1[:, 0].min() >= 0):
    #         images_and_vectors_list = variation_list_for_one_image(im, v1, sample_image_size)
    #         for varietion in images_and_vectors_list:
    #             vecs_dataset.append(varietion[1])
    #             cv2.imwrite(directory_path + '/TG_pose_im_' + str(index) + '.jpg', varietion[0])
    #             index += 1
    #     print counter2

    # counter3 = 0
    # # lspet dataset:
    # for v2 in vecs2:
    #     v2 = np.array(v2)
    #     counter3 += 1
    #     im = cv2.imread('lspet_dataset/images/im' + str(counter3).zfill(5) + '.jpg')
    #     if (im is not None) and \
    #             (v2[:, 1].max() <= im.shape[0] and v2[:, 0].max() <= im.shape[1]) and \
    #             (v2[:, 1].min() >= 0 and v2[:, 0].min() >= 0):
    #         images_and_vectors_list = variation_list_for_one_image(im, v2, sample_image_size)
    #         for varietion in images_and_vectors_list:
    #             vecs_dataset.append(varietion[1])
    #             cv2.imwrite(directory_path + '/TG_pose_im_' + str(index) + '.jpg', varietion[0])
    #             index += 1
    #     print counter3



    # writing HDF5 data:
    with h5py.File(joints_list_filename + '.hdf5', 'w') as hf:
        hf.create_dataset('joints_dataset', data=np.array(vecs_dataset).transpose((1, 2, 0)))

    print len(vecs_dataset)
    print str(sys.getsizeof(vecs_dataset)/1000) + ' MB'

    # return vecs_dataset


def load_data(portion):
    '''
    :return:
    '''

    dataset_directory_name = 'TG_pose_dataset'
    joints_list_filename = 'TG_pose_joints'
    current_directory_name = os.getcwd()
    directory_path = current_directory_name + '/' + dataset_directory_name
    joints_data = h5py.File(joints_list_filename + '.hdf5')['joints_dataset']
    only_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    print 'length of file dataset is: ' + str(len(only_files))
    images = []
    joints = []

    # only_files = only_files[:50000]
    p0 = int(portion[0]*len(only_files))
    p1 = int(portion[1]*len(only_files))

    for file_name in only_files[p0: p1]:
        image_No = int(file_name.split('_')[-1].split('.')[0])
        # print image_No
        joints_location_vector = joints_data[:, :, image_No]
        # print joints_location_vector
        image = cv2.imread(dataset_directory_name + '/' + file_name, 1)
        # plot_image_skeleton_for_testing(image, joints_location_vector)
        joints_location_vector = joints_location_vector[:, :2] #* image.shape[0]
        joints_location_vector = joints_location_vector.reshape(joints_location_vector.size)
        joints.append(joints_location_vector)
        images.append(image)

    images = np.array(images, dtype='float32') / 255
    joints = np.array(joints, dtype='float32') / images[0].shape[0]

    print 'dataset size is: ' + str(sys.getsizeof(images)/1000000 + sys.getsizeof(joints)/1000000) + ' MB'
    return images, joints


def mini_batch(portion, testing_amount):

    images, joints = load_data(portion)
    # reshaping for requiered input shape:
    images = np.transpose(images, (0, 3, 1, 2))
    data_length = len(joints)

    X_train = images[:int(1 - testing_amount * data_length)]
    Y_train = joints[:int(1 - testing_amount * data_length)]
    X_test = images[int(1 - testing_amount * data_length):]
    Y_test = joints[int(1 - testing_amount * data_length):]

    print Y_train.shape, Y_test.shape
    # print X_train.shape
    #
    # print Y_test.shape
    # print X_test.shape

    return (X_train, Y_train), (X_test, Y_test)


def train_net():
    '''
    :return:
    '''

    model_description = 'pose_model_weights'
    size_batch = 32#images[0].shape[-1]
    print size_batch

    dataset_Nof_steps = 1

    epoches_number = 10000
    overwrite_weights = True
    testing_amount = 0.1833
    Alpha = 0.3
    fully_connected_layer_size = 2**12
    (X_train, Y_train), (X_test, Y_test) = mini_batch((0.0, 0.1), testing_amount)
    max_shape_images = X_train[0].shape
    max_shape_joints = Y_train[0].shape
    print max_shape_images, max_shape_joints

    input_img = Input(shape=max_shape_images)
    # model = Sequential()
    # model.add(BatchNormalization())
    conv = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
#    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
#    residual1 = merge([input_img, conv], mode='concat', concat_axis=1)
#    conv = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(residual1)
    conv = MaxPooling2D((4, 4), strides=(4, 4))(conv)
    conv = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(4, 4))(conv)
    conv = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
#    residual2 = merge([input_img, conv], mode='concat', concat_axis=1)
#    conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)#(residual2)
#    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(1, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # mp2 = MaxPooling2D((2, 2), strides=(2, 2))
    # mp4 = MaxPooling2D((4, 4), strides=(4, 4))
    # do2D = Dropout(0.25)
    # do1D = Dropout(0.5)

    fc = Flatten()(conv)
#    fc = Dense(max_shape_joints[0], activation='relu')(fc)
#    fc = Dense(max_shape_joints[0], activation='relu')(fc)
#    fc = Dense(max_shape_joints[0], activation='relu')(fc)
    fc = Dense(max_shape_joints[0], activation='linear')(fc)
    model = Model(input=input_img, output=fc)
    model.summary()

    optimizer_method = 'adam'#SGD(lr=5e-6, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()#Adadelta()#
    model.compile(loss='mse', optimizer=optimizer_method, metrics=['accuracy'])

    ############################################################################################
    # # if previus file exist:
    # if os.path.isfile(model_description + '.hdf5'):
    #     print 'loading weights file: ' + os.path.join(model_description + '.hdf5')
    #     model.load_weights(model_description + '.hdf5')

    ############################################################################################

    EarlyStopping(monitor='val_loss', patience=0, verbose=1) #monitor='val_acc'
    checkpointer = ModelCheckpoint(model_description + '.hdf5', verbose=1, save_best_only=True)

    # model.fit(X_train, Y_train, batch_size=size_batch, nb_epoch=epoches_number,
    #                 validation_split=testing_amount, show_accuracy=True,
    #                 shuffle=True, callbacks=[checkpointer])

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    for step in range(dataset_Nof_steps):
        p0 = 1. * step / dataset_Nof_steps
        p1 = p0 + 1. / dataset_Nof_steps
        portion = (p0, p1)
        (X_train, Y_train), (X_test, Y_test) = mini_batch(portion, testing_amount)
        datagen.fit(X_train)
        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train, shuffle=True, batch_size=size_batch),
                            nb_epoch=epoches_number, verbose=1, validation_data=(X_test, Y_test),
                            callbacks=[checkpointer], class_weight=None, max_q_size=10, samples_per_epoch=len(X_train))
        # model.train_on_batch(X_train, Y_train)
        # model.test_on_batch(X_test, Y_test)

    model.save_weights(model_description + '.hdf5', overwrite_weights)





# def human_body_area_histograms(image, joints_location_vector)
#
#     # TODO: optimize so will appear only once in the code...
#     human_skeleton_masks = human_body_probability_map(image, joints_location_vector)
#
#     x = False
#     # TODO: spatiogramed based on limb and location on each, to create a generic human spatiogram, to feed logical clothing classifier and fitting vector (finger-print)
#     return x




# datasets upload:
# sample_image_size = (150, 150)
# # lsp (sport pose dataset - 2K samples):
# vecs1 = sio.loadmat('lsp_dataset/joints.mat')['joints']
# vecs1 = np.transpose(vecs1, (2, 1, 0)).astype('int')
# # lspet (extreme sport pose dataset - 10K samples):
# vecs2 = sio.loadmat('lspet_dataset/joints.mat')['joints']
# vecs2 = np.transpose(vecs2, (2, 0, 1)).astype('int')
# # fashionista (paperdoll pose dataset - 685 samples):
# vecs3 = h5py.File('fashionista_dataset/fashionista_joints.hdf5')['joints']
# vecs3 = np.transpose(vecs3, (2, 0, 1)).astype('int')
#
# print vecs1.shape, vecs2.shape, vecs3.shape
# imNo = 444
# joints_location_vector = vecs3[imNo-1]
# print joints_location_vector
#
# original_image = cv2.imread('fashionista_dataset/images/im' + str(imNo).zfill(3) + '.jpg')
# mask, grabbed_human_image = human_body_cut(original_image.copy(), joints_location_vector)
# img = plot_image_skeleton_for_testing(original_image.copy(), joints_location_vector)
# #
# # cv2.destroyAllWindows()
# #
# # superpixel_mask = superpixel_masks_parsing(original_image.copy(), mask)
# # print superpixel_mask.min(), superpixel_mask.max()
# # # regions = regionprops(superpixel_mask)
# #
# # label_rgb = np.array(label2rgb(superpixel_mask, image=original_image.copy(), bg_label=0) * 255, dtype='uint8')
# # print label_rgb.min(), label_rgb.max()
# parsed_image = color_reagion_mask(original_image.copy(), mask)
# cv2.imshow('l', np.hstack([original_image, img, grabbed_human_image, parsed_image]))
# cv2.waitKey(0)

# import_and_merge_datasets()

train_net()


