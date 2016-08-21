#this is a deploy

import cv2
import os
import numpy as np
import h5py
from skimage.segmentation import slic, quickshift
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.layers import merge, Input, Dense, Dropout, Activation, Flatten, Reshape, Convolution2D, Lambda, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2, activity_l1l2
from keras.utils import np_utils
# from sklearn.cluster import MiniBatchKMeans#, k_means
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def plot_image_skeleton_for_testing(image0, joints_location_vector):

    joints_location_vector = joints_location_vector.astype('int')

    image = image0.copy()
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

    cv2.imshow('p', image)
    cv2.waitKey(0)
    return image


def pose_net(input_shape=(3, 128, 128)):
    model_description = 'pose_lite_model_weights_No2'
    # -----------------
    # Net META parameters:
    main_Nkernels = 32
    Nmain_conv_layers = 1
    Nmain_fc_neurons = 2 ** 12
    act = 'relu'
    # -----------------

    input_img = Input(shape=(3, 128, 128))
    conv = BatchNormalization()(input_img)

    conv = Convolution2D(main_Nkernels, 7, 7, activation=act, border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    for mono_conv_layer in range(Nmain_conv_layers):
        conv = Convolution2D(main_Nkernels, 3, 3, activation=act, border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    while main_Nkernels / 2 > 15:
        main_Nkernels = main_Nkernels / 2
        conv = Convolution2D(main_Nkernels, 3, 3, activation=act, border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Convolution2D(14, 1, 1, activation='sigmoid', border_mode='same')(conv)

    fc = Flatten()(conv)
    fc = Dense(Nmain_fc_neurons, activation=act)(fc)
    fc = Dense(Nmain_fc_neurons, activation=act)(fc)
    fc = Dense(Nmain_fc_neurons / 4, activation=act)(fc)
    fc = Dense(Nmain_fc_neurons / 16, activation=act)(fc)
    fc = Dense(28, activation='linear')(fc)

    model = Model(input=input_img, output=fc)

    optimizer_method = 'adam'
    model.compile(loss='mae', optimizer=optimizer_method, metrics=['accuracy'])
    model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)), model_description + '.hdf5'))

    return model


def fit_to_net_input_size(image, max_shape=(128, 128)):
    '''
    :param image:
    :param output_size:
    :param joints_location_vector:
    :return:
    '''

    output_images_size = image.shape[:2]
    scale = 1.
    if min(output_images_size) > 1:
        # shrinking oversized images:
        if image.shape[0] > max_shape[0] or image.shape[1] > max_shape[1]:
            if image.shape[0] > image.shape[1]:
                scale = 1.0 * max_shape[0] / image.shape[0]
            else:
                scale = 1.0 * max_shape[1] / image.shape[1]

            image = cv2.resize(image.copy(), (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            output_images_size = image.shape[:2]

        h_max, w_max = max_shape  # output_size
        h, w = output_images_size
        w_shift = w_max - w
        h_shift = h_max - h

        output_image = cv2.copyMakeBorder(image.copy(), h_shift / 2, h_shift / 2, w_shift / 2, w_shift / 2, cv2.BORDER_REPLICATE)
        output_image_0 = output_image.copy()
        output_image = cv2.resize(output_image_0, max_shape)

    return output_image, scale, h_shift, w_shift


def find_pose(image0, model):
    ###########
    # reshape input image:
    original_image_shape = image0.shape
    fitted_image, scale, h_shift, w_shift = fit_to_net_input_size(image0)
    size_of_image = fitted_image.shape
    image1 = fitted_image.reshape((1, size_of_image[0], size_of_image[1], size_of_image[2]))
    image1 = np.array(image1, dtype='float32') / 255
    image1 = np.transpose(image1, (0, 3, 1, 2))
    prediction = model.predict(image1, batch_size=1)

    joints = prediction.reshape(prediction.shape[1] / 2, 2) * fitted_image.shape[-2]
    joints[:, 0] = joints[:, 0] - w_shift/2
    joints[:, 1] = joints[:, 1] - h_shift/2
    joints = joints / scale
    ones = np.ones((joints.shape[0], 1))
    joints_location_vector = np.hstack([joints, ones])
    return joints_location_vector


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


def superpixel_masks_parsing(image, mask):

    # bbox for speed slicing:
    y0, x0, dy, dx = cv2.boundingRect(mask)
    number_of_segments = (dx*dy)/((dx+dy)/4)
    # print number_of_segments
    image2 = (image.copy() * mask[:, :, np.newaxis])[x0:x0+dx, y0:y0+dy]

    # apply SLIC and extract (approximately) the supplied number of segments:
    segments = slic(image2, n_segments=number_of_segments, compactness=5, max_iter=13,
                    sigma=13, spacing=None, multichannel=True,
                    convert2lab=True, ratio=None)

    # # apply QUICKSHIFT and extract (approximately) the supplied number of segments:
    # segments = quickshift(image2, max_dist=np.sqrt(number_of_segments), sigma=0.1, convert2lab=True)

    segments = segments + 1 # So that no labelled region is 0 and ignored by regionprops
    segments = segments * mask[x0:x0+dx, y0:y0+dy]
    superpixel_msk = mask.astype('int')
    superpixel_msk[x0:x0+dx, y0:y0+dy] = segments

    # print superpixel_msk.min(), superpixel_msk.max()
    return superpixel_msk


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


def cut_by_superpixels_and_limb_cojunction(image, joints_location_vector):

    limb_masks = human_body_probability_map(image, joints_location_vector)
    #whole_body, rbl, rul, pelvic, lul, lbl, rba, rua, shoul, neck, lba, lua, head, td1, td2, ut, lt, ft

    mask, grabbed_human_image = human_body_cut(image.copy(), joints_location_vector)

    superpixel_msk = superpixel_masks_parsing(image, mask)

    limb_superpixeled_masks = []
    for limb in limb_masks:
        sp_msk = np.unique(superpixel_msk.copy()[limb>0])
        sp_msk = sp_msk[sp_msk>0]
        limb_mask = np.zeros(mask.shape)
        for i in range(len(sp_msk)):
            limb_mask[np.where(superpixel_msk==sp_msk[i])] = 1
        grabbed_limb_image = image.copy()
        grabbed_limb_image[limb_mask==0] = 0
        limb_superpixeled_masks.append(grabbed_limb_image)

    return limb_masks, limb_superpixeled_masks, mask, grabbed_human_image



image = cv2.imread('/home/nate/Desktop/8c30a2582a64434c87fe0c504e2c1640.jpg')
print image.shape
model = pose_net()
joints_location_vector = find_pose(image, model)
plot_image_skeleton_for_testing(image, joints_location_vector)

