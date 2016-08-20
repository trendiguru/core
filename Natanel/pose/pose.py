import cv2
import numpy as np
import os
import sys
import scipy.io as sio
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


###################################################################
# visualization:

def plot_image_skeleton_for_testing(image, joints_location_vector):

    joints_location_vector = joints_location_vector.astype('int')

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


def visualize_pose_on_dataset(path_to_images, path_to_joints_hdf5):
    vecs = h5py.File(path_to_joints_hdf5)['dataset_joints']
    for i in range(96):
        im = cv2.imread(path_to_images + '/TG_pose_im_' + str(i) + '.jpg')
        v = vecs[:, :, i]
        plot_image_skeleton_for_testing(im, v)

###################################################################
# dataset acquisition:
numeric_type_of_joints_coordinates = 'float32'

def load_lsp_dataset():
    lsp_dataset_list = []
    # lsp (sport pose dataset - 2K samples):
    vecs1 = sio.loadmat('lsp_dataset/joints.mat')['joints']
    vecs1 = np.transpose(vecs1, (2, 1, 0)).astype(numeric_type_of_joints_coordinates)
    counter = 0
    # lsp dataset:
    for v1 in vecs1:
        v1 = np.array(v1)
        if not (v1[:, :2] <= 0).any():
            counter += 1
            image_path = 'lsp_dataset/images/im' + str(counter).zfill(4) + '.jpg'
            lsp_dataset_list.append([image_path, v1])

    return lsp_dataset_list


def load_lspet_dataset():
    lspet_dataset_list = []
    # lspet (extreme sport pose dataset - 10K samples):
    vecs2 = sio.loadmat('lspet_dataset/joints.mat')['joints']
    vecs2 = np.transpose(vecs2, (2, 0, 1)).astype(numeric_type_of_joints_coordinates)
    counter = 0
    # lspet dataset:
    for v2 in vecs2:
        v2 = np.array(v2)
        if not (v2[:, :2] <= 0).any():
            counter += 1
            image_path = 'lspet_dataset/images/im' + str(counter).zfill(5) + '.jpg'
            lspet_dataset_list.append([image_path, v2])

    return lspet_dataset_list


def load_fashionista_dataset():
    fashionista_dataset_list = []
    # fashionista (paperdoll pose dataset - 685 samples):
    vecs3 = h5py.File('fashionista_dataset/fashionista_joints.hdf5')['joints']
    vecs3 = np.transpose(vecs3, (2, 0, 1)).astype(numeric_type_of_joints_coordinates)
    counter = 0
    # fashionista dataset:
    for v3 in vecs3:
        v3 = np.array(v3)
        if not (v3[:, :2] <= 0).any():
            counter += 1
            image_path = 'fashionista_dataset/images/im' + str(counter).zfill(3) + '.jpg'
            fashionista_dataset_list.append([image_path, v3])

    return fashionista_dataset_list


def mpii_dataset_joint_standart_interpreter(index):

    # Right_ankle = flipped_joints_location_vector[0, 0]
    # Right_knee = flipped_joints_location_vector[1, 0]
    # Right_hip = flipped_joints_location_vector[2, 0]
    # Left_hip = flipped_joints_location_vector[3, 0]
    # Left_knee = flipped_joints_location_vector[4, 0]
    # Left_ankle = flipped_joints_location_vector[5, 0]
    # Right_wrist = flipped_joints_location_vector[6, 0]
    # Right_elbow = flipped_joints_location_vector[7, 0]
    # Right_shoulder = flipped_joints_location_vector[8, 0]
    # Left_shoulder = flipped_joints_location_vector[9, 0]
    # Left_elbow = flipped_joints_location_vector[10, 0]
    # Left_wrist = flipped_joints_location_vector[11, 0]
    # Neck = joints_location_vector[12, :]
    # Head_top = joints_location_vector[13, :]
    #
    # (0 - r ankle,
    #  1 - r knee,
    #  2 - r hip,
    #  3 - l hip,
    #  4 - l knee,
    #  5 - l ankle,
    #  6 - pelvis,
    #  7 - thorax,
    #  8 - upper neck,
    #  9 - head top,
    #  10 - r wrist,
    #  11 - r elbow
    #  12 - r shoulder,
    #  13 - l shoulder,
    #  14 - l elbow,
    #  15 - l wrist)"
    if 0<=index<=5:
        lsp_standard_index = index
    elif 10<=index<=15:
        lsp_standard_index = index-4
    elif index==8:
        lsp_standard_index = 12
    elif index==9:
        lsp_standard_index = 13
    else:
        lsp_standard_index = 666

    return lsp_standard_index
def rearange_mpii_joints_by_standard(joints_vector):
    lsp_standard_joints_vector = np.zeros((14, 3), dtype=numeric_type_of_joints_coordinates)
    for line in joints_vector:
        joint_by_mpii_standard = line[0]
        lsp_standard_index = mpii_dataset_joint_standart_interpreter(joint_by_mpii_standard)
        if lsp_standard_index != 666:
            # if lsp_standard_index==13:
            #     line[1:][:2] = (line[1:][:2] + lsp_standard_joints_vector[12, :2])/2
            lsp_standard_joints_vector[lsp_standard_index, :] = line[1:]
    return lsp_standard_joints_vector
def load_mpii_dataset():
    vecs4 = sio.loadmat('mpii_dataset/mpii_human_pose_v1_u12_1.mat')['RELEASE'][0]['annolist'][0][0]
    mpii_dataset_list = []
    for v4 in vecs4:
        image_path = 'mpii_dataset/images/' + v4['image'][0][0][0][0]
        annorect = v4['annorect']
        # print image_path
        if len(annorect)>0:
            for rect in annorect[0]:
                try:
                    try:
                        try:
                            X_points = np.array(rect['annopoints'][0]['point'][0][0]['x'], dtype=numeric_type_of_joints_coordinates)
                            Y_points = np.array(rect['annopoints'][0]['point'][0][0]['y'], dtype=numeric_type_of_joints_coordinates)
                            id_points = np.array(rect['annopoints'][0]['point'][0][0]['id'], dtype=numeric_type_of_joints_coordinates)
                            F = rect['annopoints'][0]['point'][0][0]['is_visible']
                            K = []
                            for f in F:
                                if len(f) == 0:
                                    k = 0
                                else:
                                    k = int(f.astype('uint8'))#int(f.flatten())
                                K.append(k)
                            is_visible_points = np.array(K, dtype=numeric_type_of_joints_coordinates)
                            X_points = X_points.reshape((1, len(X_points)))
                            Y_points = Y_points.reshape((1, len(Y_points)))
                            id_points = id_points.reshape((1, len(id_points)))
                            is_visible_points = is_visible_points.reshape((1, len(is_visible_points)))
                            vr4 = np.hstack((id_points.T, X_points.T, Y_points.T, is_visible_points.T))
                            vr4 = rearange_mpii_joints_by_standard(vr4)
                            if not (vr4[:, :2] <= 0).any():
                                mpii_dataset_list.append([image_path, vr4])
                                # print vr4
                        except TypeError:
                            pass
                    except IndexError:
                        pass
                except ValueError:
                    pass

    return mpii_dataset_list

# # TODO: fix buffy dataset...
# def buffy_dataset():
#     # get text files vs images folders...

# # TODO: fix video pose dataset...
# def video_pose_dataset():
#     vecs6 = sio.loadmat('video_pose_dataset/clips.mat')['clips'][0]
#     for scene in vecs6:
#         for image in scene[4][0]:
#             image_name = image[6]#['name']
#             bbox = image[5]#???
#             points = image[4]
#
#     print image_name
#     print points

def merge_datasets_to_lists(validation_fraction):

    if validation_fraction > 0.5:
        print 'validation_fraction should be less than 0.5 (50%) of dataset.'
        return None
    lsp_list = load_lsp_dataset()
    lspet_list = load_lspet_dataset()
    fashionista_list = load_fashionista_dataset()
    mpii_list = load_mpii_dataset()

    training_list = lsp_list[:int((1-validation_fraction)*len(lsp_list))] + \
                    lspet_list[:int((1-validation_fraction)*len(lspet_list))] + \
                    fashionista_list[:int((1-validation_fraction)*len(fashionista_list))] + \
                    mpii_list[:int((1-validation_fraction)*len(mpii_list))]

    validation_list = lsp_list[int((1-validation_fraction)*len(lsp_list)):] + \
                      lspet_list[int((1-validation_fraction)*len(lspet_list)):] + \
                      fashionista_list[int((1-validation_fraction)*len(fashionista_list)):] + \
                      mpii_list[int((1-validation_fraction)*len(mpii_list)):]

    print 'training_list length (samples): ',  len(training_list)
    print 'validation_list length (samples): ', len(validation_list)
    return training_list, validation_list


###################################################################
# dataset creation:

def pose_bbox(joints_location_vector):
    '''
    :param joints_location_vector:
    :return:
    '''
    upper_left_corner = np.array([joints_location_vector[:, 0].min(), joints_location_vector[:, 1].min()])
    upper_left_corner[upper_left_corner<0] = 0
    lower_right_corner = np.array([joints_location_vector[:, 0].max(), joints_location_vector[:, 1].max()])
    lower_right_corner[lower_right_corner<0] = 0

    # dXbbox, dYbbox = lower_right_corner - upper_left_corner
    # W = (dYbbox**2 + dXbbox**2)**0.5
    # if W >= 100:
    #     return upper_left_corner, lower_right_corner
    # else:
    #     print ' body bbox diagonal is <100p...'
    #     return None
    return upper_left_corner, lower_right_corner


def enlarged_pose_bbox(joints_location_vector):
    upper_left_corner, lower_right_corner = pose_bbox(joints_location_vector)
    dXbbox, dYbbox = lower_right_corner - upper_left_corner
    margine = int(((dYbbox**2 + dXbbox**2)**0.5)/9)
    enlarged_upper_left_corner = upper_left_corner - (margine, margine)
    enlarged_upper_left_corner[enlarged_upper_left_corner<0] = 0
    enlarged_lower_right_corner = lower_right_corner + (margine, margine)
    enlarged_lower_right_corner[enlarged_lower_right_corner<0] = 0
    return enlarged_upper_left_corner, enlarged_lower_right_corner


def bbox2img_ratio(image, joints_location_vector):

    shape = image.shape[:2]
    upper_left_corner, lower_right_corner = pose_bbox(joints_location_vector)
    dXbbox, dYbbox = lower_right_corner - upper_left_corner
    margine = ((dYbbox**2 + dXbbox**2)**0.5)/9
    dYbbox += margine
    dXbbox += margine
    ratioY = dYbbox / shape[0]
    ratioX = dXbbox / shape[1]
    print ratioY, ratioX
    # im = cv2.rectangle(image, tuple(upper_left_corner-int(margine)), tuple(lower_right_corner+int(margine)), (0, 0, 255), thickness=2,  lineType=8, shift=0)
    # cv2.imshow('p', im)
    # cv2.waitKey(0)
    return ratioY, ratioX


def rotation_matrix(angle):
    # angle in degs!
    theta = (angle/180.)*np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    return rotMatrix


def pose_center_point_for_rotation(joints_location_vector):
    upper_left_corner, lower_right_corner = pose_bbox(joints_location_vector)
    center_point_for_rotation = (upper_left_corner + lower_right_corner) / 2
    return center_point_for_rotation


# def rotate_image(image, point_of_rotation, angle):
#     border_type = cv2.BORDER_REPLICATE
#     row, col, dep = image.shape
#     ncol = int(np.cos(np.deg2rad(angle)) * row + np.sin(np.deg2rad(angle)) * col)
#     nrow = int(np.cos(np.deg2rad(angle)) * col + np.sin(np.deg2rad(angle)) * row)
#
#     # # We require a translation matrix to keep the image centred:
#     # translation_mat = np.matrix([
#     #     [1, 0, int(ncol * 0.5 - col/2)],
#     #     [0, 1, int(nrow * 0.5 - row/2)],
#     #     [0, 0, 1]])
#
#     rotated_image_matrix = cv2.getRotationMatrix2D(tuple(point_of_rotation), -angle, 1.0)
#
#     # # Convert the OpenCV 3x2 rotation matrix to 3x3:
#     # rotation_mat = np.vstack([rotated_image_matrix, [0, 0, 1]])
#     #
#     # # Compute the tranform for the combined rotation and translation
#     # affine_mat = (np.matrix(translation_mat) * np.matrix(rotation_mat))[0:2, :]
#
#     rotated_image = cv2.warpAffine(image, affine_mat, (nrow, ncol), borderMode=border_type)
#     cv2.imshow('p', rotated_image)
#     cv2.waitKey(0)
#     return rotated_image


def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def rotate_image(image, point_of_rotation, angle):
    """
    Rotates the given image about it's centre
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(point_of_rotation)#np.array(image_size) / 2)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, -angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, borderMode=cv2.BORDER_REPLICATE)

    return result



def rotate_joints(joints_location_vector, point_of_rotation, angle):
    RM = rotation_matrix(angle)
    rotated_joints_location_vector = joints_location_vector
    for i in range(joints_location_vector.shape[0]):
        rot = np.dot(RM, [joints_location_vector[i, 0] - point_of_rotation[0],
                          joints_location_vector[i, 1] - point_of_rotation[1]])
        rotated_joints_location_vector[i, 0] = rot[0] + point_of_rotation[0]
        rotated_joints_location_vector[i, 1] = rot[1] + point_of_rotation[1]
    return rotated_joints_location_vector


def vflipped_joints_vectors(image, joints_location_vector):
    '''
    :param image:
    :param joints_location_vector:
    :return:
    '''

    shape = image.shape
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

    return flipped_joints_location_vector


def color_delta(image, delta=30):

    images = [image]
    try:
        if not (np.array(image.shape[:2]) == 0).any():
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
    except AttributeError:
        pass

    return images


def fit_to_net_input_size(image, joints_location_vector, max_shape=(192, 192)):
    '''
    :param image:
    :param output_size:
    :param joints_location_vector:
    :return:
    '''

    shifting = True
    output_images_size = image.shape[:2]
    # shrinking oversized images:
    if image.shape[0] > max_shape[0] or image.shape[1] > max_shape[1]:
        # image, joints_location_vector = crop_enlarged_bbox(image.copy(), joints_location_vector.copy())
        if image.shape[0] > image.shape[1]:
            # output_images_size = (image.shape[1]*max_shape[0]/image.shape[0], max_shape[0])
            scale = 1.0 * max_shape[0] / image.shape[0]
        else:
            # output_images_size = (max_shape[1], image.shape[0]*max_shape[1]/image.shape[1])
            scale = 1.0 * max_shape[1] / image.shape[1]

        joints_location_vector[:, 0] = joints_location_vector[:, 0] * scale
        joints_location_vector[:, 1] = joints_location_vector[:, 1] * scale
        image = cv2.resize(image.copy(), max_shape)
        output_images_size = image.shape[:2]

    border_type = cv2.BORDER_REPLICATE
    # getting image size:
    h_max, w_max = max_shape  # output_size
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
    # no rotate:
    location_vector = joints_location_vector.copy()
    output_image = cv2.copyMakeBorder(image.copy(), h_shift / 2, h_shift / 2, w_shift / 2, w_shift / 2, border_type)
    location_vector[:, 0] = joints_location_vector[:, 0] + w_shift / 2
    location_vector[:, 1] = joints_location_vector[:, 1] + h_shift / 2
    joints_location_vector_0 = location_vector.astype(int)
    output_image_0 = output_image.copy()
    output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
    # print joints_location_vector_0
    ####################################################################
    # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
    ###################################################################

    if shifting:
        if cornerize1:
            # 1: upper left corner placement
            # no rotate:
            location_vector = joints_location_vector.copy()
            output_image = cv2.copyMakeBorder(image.copy(), 0, h_shift, 0, w_shift, border_type)
            location_vector[:, 0] = joints_location_vector[:, 0]
            location_vector[:, 1] = joints_location_vector[:, 1]
            joints_location_vector_0 = location_vector.astype(int)
            output_image_0 = output_image.copy()
            output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
            # print joints_location_vector_0
            ####################################################################
            # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
            ###################################################################

            if w_shift > w_max / 3:
                # 2: lower left corner placement
                # no rotate:
                location_vector = joints_location_vector.copy()
                output_image = cv2.copyMakeBorder(image.copy(), h_shift, 0, 0, w_shift, border_type)
                location_vector[:, 0] = joints_location_vector[:, 0]
                location_vector[:, 1] = joints_location_vector[:, 1] + h_shift
                joints_location_vector_0 = location_vector.astype(int)
                output_image_0 = output_image.copy()
                output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
                # print joints_location_vector_0
                ####################################################################
                # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
                ###################################################################

        if cornerize2:
            # 3: upper right corner placement
            # no rotate:
            location_vector = joints_location_vector.copy()
            output_image = cv2.copyMakeBorder(image.copy(), 0, h_shift, w_shift, 0, border_type)
            location_vector[:, 0] = joints_location_vector[:, 0] + w_shift
            location_vector[:, 1] = joints_location_vector[:, 1]
            joints_location_vector_0 = location_vector.astype(int)
            output_image_0 = output_image.copy()
            output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
            # print joints_location_vector_0
            ####################################################################
            # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
            ###################################################################

            # 4: lower right corner placement
            if h_shift > h_max / 3:
                # no rotate:
                location_vector = joints_location_vector.copy()
                output_image = cv2.copyMakeBorder(image.copy(), h_shift, 0, w_shift, 0, border_type)
                location_vector[:, 0] = joints_location_vector[:, 0] + w_shift
                location_vector[:, 1] = joints_location_vector[:, 1] + h_shift
                joints_location_vector_0 = location_vector.astype(int)
                output_image_0 = output_image.copy()
                output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
                # print joints_location_vector_0
                ####################################################################
                # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
                ###################################################################

    # sizing up small images:
    if (cornerize1 or cornerize2) and min(1.0 * w_shift / w_max, 1.0 * h_shift / h_max) > 1. / 3:
        if w_shift > h_shift:
            # output_images_size = (w * h_max / h, h_max)
            scale = 1.0 * h_max / h
            w_shift = w_max - w * h_max / h
            h_shift = 0
        else:
            # output_images_size = (w_max, h * w_max / w)
            scale = 1.0 * w_max / w
            w_shift = 0
            h_shift = h_max - h * w_max / w

        joints_location_vector_d = joints_location_vector.copy()
        image = cv2.resize(image.copy(), max_shape)
        joints_location_vector_d[:, 0] = joints_location_vector[:, 0] * scale
        joints_location_vector_d[:, 1] = joints_location_vector[:, 1] * scale

        # 0: central placement
        # no rotate:
        location_vector = joints_location_vector_d.copy()
        output_image = cv2.copyMakeBorder(image.copy(), h_shift / 2, h_shift / 2, w_shift / 2, w_shift / 2, border_type)
        location_vector[:, 0] = joints_location_vector_d[:, 0] + w_shift / 2
        location_vector[:, 1] = joints_location_vector_d[:, 1] + h_shift / 2
        joints_location_vector_0 = location_vector.astype(int)
        output_image_0 = output_image.copy()
        output_images_and_vectors.append([cv2.resize(output_image_0, max_shape), joints_location_vector_0])
        # print joints_location_vector_0
        ####################################################################
        # plot_image_skeleton_for_testing(output_image_0, joints_location_vector_0)
        ###################################################################

    return output_images_and_vectors


def dataset_creation(dataset_name_str, dataset_list, number_of_angles_to_rotate=20):
    # dataset list is in shape: [image_path, joints_vector]

    # output list of joints file:
    joints_list_filename = dataset_name_str + '_joints'
    # creating a directory for the dataset images (augmented) and their pose (joints points) vectors:
    dataset_directory_name = dataset_name_str + '_images'
    # current_directory_name = os.getcwd()
    # directory_path = current_directory_name + '/' + dataset_directory_name
    if not os.path.exists(dataset_directory_name):
        os.mkdir(dataset_directory_name)

    delta_angle = 360/number_of_angles_to_rotate
    angles = range(0, 360, delta_angle)
    index = 0
    vecs_dataset = []
    percent_done0 = 0

    dataset_length = len(dataset_list)
    sample_count = 0
    for sample in dataset_list:
        # colorizing:
        images = color_delta(cv2.imread(sample[0]))
        vector = sample[1]
        p = pose_center_point_for_rotation(vector) # rotation point
        for image in images:
            try:
                if not (np.array(image.shape[:2])==0).any():
                    # rotating:
                    for angle in angles:
                        im0 = rotate_image(image.copy(), p, angle)
                        vec = rotate_joints(vector.copy(), p, angle)
                        ###
                        vec[:, 0] = vec[:, 0] + (im0.shape[1] - image.shape[1])/2
                        vec[:, 1] = vec[:, 1] + (im0.shape[0] - image.shape[0])/2
                        ###
                        upper_left_corner, lower_right_corner = enlarged_pose_bbox(vec)
                        # cropping the single person box / image:
                        #
                        # plot_image_skeleton_for_testing(im0.copy(), vec)
                        #
                        im1 = im0[upper_left_corner[1]:lower_right_corner[1],
                                  upper_left_corner[0]:lower_right_corner[0], :]
                        vec[:, 0] = vec[:, 0] - upper_left_corner[0]
                        vec[:, 1] = vec[:, 1] - upper_left_corner[1]
                        #
                        # plot_image_skeleton_for_testing(im1.copy(), vec)
                        #
                        if not (np.array(im1.shape[:2])==0).any():
                            images_and_vectors_list = fit_to_net_input_size(im1, vec)
                            for varietion in images_and_vectors_list:
                                if len(varietion)>0:
                                    # unflipped:
                                    vecs_dataset.append(varietion[1])
                                    cv2.imwrite(dataset_directory_name + '/im_' + str(index) + '.jpg', varietion[0])
                                    index += 1
                                    # flipped:
                                    vecs_dataset.append(vflipped_joints_vectors(varietion[0], varietion[1]))
                                    cv2.imwrite(dataset_directory_name + '/im_' + str(index) + '.jpg', np.fliplr(varietion[0]))
                                    index += 1
            except AttributeError:
                pass
        sample_count += 1
        percent_done = int(100.0 * sample_count / dataset_length)
        if percent_done0 < percent_done:
            percent_done0 = percent_done
            sys.stdout.write('\r' + str(index) + ' images... ' + str(sample_count) + ' / ' + str(dataset_length) +
                             ' = ' + str(100.0 * sample_count / dataset_length) + ' % done.')
            sys.stdout.flush()

    # writing HDF5 data:
    with h5py.File(joints_list_filename + '.hdf5', 'w') as hf:
        hf.create_dataset('dataset_joints', data=np.array(vecs_dataset).transpose((1, 2, 0)))

    print '\n' + str(1.0 * sys.getsizeof(vecs_dataset) / 1000) + ' MB'


def create_test_and_train_set(validation_fraction=0.1):
    training_list, validation_list = merge_datasets_to_lists(validation_fraction)
    current_directory_name = os.getcwd()
    directory_path = current_directory_name + '/TG_dataset'
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    # creating training dataset:
    print 'starting to build training dataset...'
    dataset_creation(directory_path + '/' + 'TG_training', training_list) # [:int(len(training_list)*0.005)]TODO: erase limitation
    print 'finished building training dataset.'
    # creating validation dataset:
    print 'starting to build validation dataset...'
    dataset_creation(directory_path + '/' + 'TG_validation', validation_list) # [:int(len(validation_list)*0.005)]TODO: erase limitation
    print 'finished building validation dataset...'


###################################################################
# training & neural net creation:

def load_data(dataset_directory_name, joints_list_filename, portion):
    '''
    :return:
    '''

    current_directory_name = os.getcwd()
    directory_path = current_directory_name + '/' + dataset_directory_name
    joints_data = h5py.File(joints_list_filename)['dataset_joints'] # needs to be a *.hdf5 type file...
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
        image = cv2.imread(dataset_directory_name + '/' + file_name, 0) # grayscale = 0, BGR = 1
	image = cv2.equalizeHist(image)
        # plot_image_skeleton_for_testing(image, joints_location_vector)
        joints_location_vector[:, :2] = joints_location_vector[:, :2].astype('float32') / image.shape[0]
        joints_location_vector = joints_location_vector.reshape(joints_location_vector.size)
        joints.append(joints_location_vector)
	image = image.reshape((image.shape[0], image.shape[1], 1)) # for grayscale
        images.append(image)

    images = np.array(images, dtype='float32') / 255
    joints = np.array(joints, dtype='float32')# / images[0].shape[0]
    # reshaping for requiered input shape:
    images = np.transpose(images, (0, 3, 1, 2))

    print 'dataset size is: ' + str(sys.getsizeof(images)/1000000 + sys.getsizeof(joints)/1000000) + ' MB'
    return images, joints


def mini_batch(training_dataset_directory_name, training_joints_list_filename,
               validation_dataset_directory_name, validation_joints_list_filename, portion):

    training_images, training_joints = load_data(training_dataset_directory_name, training_joints_list_filename,
                                                 portion)
    validation_images, validation_joints = load_data(validation_dataset_directory_name, validation_joints_list_filename,
                                                 portion)




    return (training_images, training_joints), (validation_images, validation_joints)


def pose_generator(dataset_directory_name, joints_list_filename):#, dataset_Nof_steps):
    while 1:
        # for step in range(dataset_Nof_steps):
        #     p0 = 1. * step / dataset_Nof_steps
        #     p1 = p0 + 1. / dataset_Nof_steps
        #     portion = (p0, p1)
        images, joints = load_data(dataset_directory_name, joints_list_filename, (0.0, 1.0))#portion)
        print images.shape, joints.shape
        yield images, joints




def train_net():
    '''
    :return:
    '''

    # dataset:
    ## training:
    training_dataset_directory_name = 'TG_dataset/TG_training_images'
    training_joints_list_filename = 'TG_dataset/TG_training_joints.hdf5'
    ## validation:
    validation_dataset_directory_name = 'TG_dataset/TG_validation_images'
    validation_joints_list_filename = 'TG_dataset/TG_validation_joints.hdf5'

    model_description = 'pose_model_weights_new'
    dataset_Nof_steps = 50
    epoches_number = 10000000
    overwrite_weights = True
    #-----------------
    # Net META parameters:
    size_batch = 32
    main_Nkernels = 32
    Nmain_conv_layers = 5
    Nmain_fc_neurons = 2**12
    act = 'relu'
    #-----------------

    (X_train, Y_train), (X_test, Y_test) = mini_batch(training_dataset_directory_name, training_joints_list_filename,
               validation_dataset_directory_name, validation_joints_list_filename, (0.0, 0.00001))
    max_shape_images = X_train[0].shape
    max_shape_joints = Y_train[0].shape
    print max_shape_images, max_shape_joints

    input_img = Input(shape=max_shape_images)
    conv = BatchNormalization()(input_img)


#    conv = Convolution2D(main_Nkernels, 11, 11, activation=act, border_mode='same')(conv)
    conv = Convolution2D(main_Nkernels, 9, 9, activation=act, border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    for mono_conv_layer in range(Nmain_conv_layers):
        main_Nkernels = main_Nkernels*2
#       if main_Nkernels > 512:
#               main_Nkernels = 512
        conv = Convolution2D(main_Nkernels, 3, 3, activation=act, border_mode='same')(conv)
#       conv = Convolution2D(main_Nkernels, 3, 3, activation=act, border_mode='same')(conv)
        conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
#        conv = Convolution2D(main_Nkernels, 5, 5, activation=act, border_mode='same')(conv)    
#    main_Nkernels = main_Nkernels*2
#    k = 3
#    while main_Nkernels/2 > 3:
#        main_Nkernels = main_Nkernels/2
#       #k += 2
#        conv = Convolution2D(main_Nkernels, k, k, activation=act, border_mode='same')(conv)
#        conv = Convolution2D(main_Nkernels, 3, 3, activation=act, border_mode='same')(conv)
#        conv = Convolution2D(main_Nkernels, 3, 3, activation=act, border_mode='same')(conv)
#    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)   
#    conv = Convolution2D(14, 1, 1, activation='sigmoid', border_mode='same')(conv)  
#    conv1 = Convolution2D(1, 128, 1, activation='tanh', border_mode='valid')(conv)
#    conv2 = Convolution2D(1, 1, 128, activation='tanh', border_mode='valid')(conv)
#    fc1 = Flatten()(conv1)
#    fc2 = Flatten()(conv2)
#    fc = Merge([fc1, fc2], mode='concat')

    fc = Flatten()(conv)
    fc = Dropout(0.25)(fc)
    ####
    fc = Dense(Nmain_fc_neurons, activation=act)(fc)
    fc = Dropout(0.5)(fc)
#    fc = Dense(Nmain_fc_neurons/32, activation=act)(fc)
#    fc = Dense(Nmain_fc_neurons/32, activation=act)(fc)
#    fc = Dense(Nmain_fc_neurons/32, activation=act)(fc)
    fc = Dense(Nmain_fc_neurons, activation=act)(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(Nmain_fc_neurons, activation=act)(fc)
    fc = Dropout(0.5)(fc)
    ####
    fc = Dense(max_shape_joints[0], activation='linear')(fc)


#    conv = Convolution2D(16, 7, 7, activation='relu', border_mode='same')(conv)
#    # residual1 = merge([input_img, conv], mode='concat', concat_axis=1)
#    # conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(residual1)
#    for C1_layers in range(16):
#        conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)
#
#    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(conv)
#    conv = Convolution2D(2, 3, 3, activation='relu', border_mode='same')(conv)
#    # mp2 = MaxPooling2D((2, 2), strides=(2, 2))
#    # mp4 = MaxPooling2D((4, 4), strides=(4, 4))
#    # do2D = Dropout(0.25)
#    # do1D = Dropout(0.5)
#
#    fc = Flatten()(conv)
#    fc = Dense(max_shape_joints[0], activation='linear')(fc)
    model = Model(input=input_img, output=fc)
    model.summary()

    optimizer_method = 'adam'#SGD(lr=5e-6, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()#Adadelta()#
    model.compile(loss='mae', optimizer=optimizer_method, metrics=['accuracy'])

    ############################################################################################
    # # if previus file exist:
    # if os.path.isfile(model_description + '.hdf5'):
    #     print 'loading weights file: ' + os.path.join(model_description + '.hdf5')
    #     model.load_weights(model_description + '.hdf5')

    ############################################################################################

    EarlyStopping(monitor='val_loss', patience=0, verbose=1) #monitor='val_acc'
    checkpointer = ModelCheckpoint(model_description + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    # model.fit_generator(pose_generator(training_dataset_directory_name, training_joints_list_filename),#, dataset_Nof_steps),
    #                     validation_data=pose_generator(validation_dataset_directory_name, validation_joints_list_filename),#, dataset_Nof_steps),
    #                     max_q_size=10, nb_epoch=2, verbose=1, callbacks=[checkpointer])#, samples_per_epoch=, nb_val_samples=
    #                     # class_weight=None, samples_per_epoch=len(X_train))


#    # this will do preprocessing and realtime data augmentation
#    datagen = ImageDataGenerator(
#        featurewise_center=False,  # set input mean to 0 over the dataset
#        samplewise_center=False,  # set each sample mean to 0
#        featurewise_std_normalization=False,  # divide inputs by std of the dataset
#        samplewise_std_normalization=False,  # divide each input by its std
#        zca_whitening=False,  # apply ZCA whitening
#        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
#        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
#        horizontal_flip=False,  # randomly flip images
#        vertical_flip=False)  # randomly flip images
#    # compute quantities required for featurewise normalization
#    # (std, mean, and principal components if ZCA whitening is applied)
 

#   for epoch_No in range(epoches_number):
#        for step in range(dataset_Nof_steps):
#            p0 = 1. * step / dataset_Nof_steps
#            p1 = p0 + 1. / dataset_Nof_steps
#            portion = (p0, p1)
#            (X_train, Y_train), (X_validation, Y_validation) = mini_batch(training_dataset_directory_name,
#                                                              training_joints_list_filename,
#                                                              validation_dataset_directory_name,
#                                                              validation_joints_list_filename,
#                                                              portion)
#            print 'step No: ', step + 1, '/', dataset_Nof_steps, '...  @ epoch No: ', epoch_No + 1
#            model.fit(X_train, Y_train, batch_size=size_batch, nb_epoch=1, verbose=1, callbacks=[checkpointer],
#                      validation_split=0.0, validation_data=(X_validation, Y_validation), shuffle=True,
#                      class_weight=None, sample_weight=None)

#    model.load_weights(model_description + '.hdf5')
    for epoch_No in range(epoches_number):
        for step in range(dataset_Nof_steps):
            p0 = 1. * step / dataset_Nof_steps
            p1 = p0 + 1. / dataset_Nof_steps
            portion = (p0, p1)
            (X_train, Y_train), (X_validation, Y_validation) = mini_batch(training_dataset_directory_name,
                                                              training_joints_list_filename,
                                                              validation_dataset_directory_name,
                                                              validation_joints_list_filename,
                                                              portion)
            print ' ### step No: ', step + 1, '/', dataset_Nof_steps, '...  @ epoch No: ', epoch_No + 1, '###'
            model.fit(X_train, Y_train, batch_size=size_batch, nb_epoch=1, verbose=1, callbacks=[checkpointer],
                      validation_split=0.0, validation_data=(X_validation, Y_validation), shuffle=True,
                      class_weight=None, sample_weight=None)


            # # TODO: ImageDataGenerator + numpy fuckup - needs to be resolved...
            # model.fit(X_train, Y_train, batch_size=size_batch, nb_epoch=1, validation_split=0.0,
            #           validation_data=(X_test, Y_test), shuffle=True, callbacks=[checkpointer])
            # datagen.fit(X_train)
            # fit the model on the batches generated by datagen.flow()
            # model.fit_generator(datagen.flow(X_train, Y_train, shuffle=True, batch_size=size_batch),
            #                     nb_epoch=1, verbose=1, validation_data=(X_test, Y_test),
            #                     callbacks=[checkpointer], class_weight=None, max_q_size=10, samples_per_epoch=len(X_train))
            
            # model.load_weights(model_description + '.hdf5')
            
            # model.train_on_batch(X_train, Y_train)
            # model.test_on_batch(X_test, Y_test)

    model.save_weights(model_description + '.hdf5', overwrite_weights)






# merge_datasets_to_lists(0.1833)

# list = load_mpii_dataset()
#
# imNo = 878
# dataset_creation('TG', [list[imNo]])
# im = cv2.imread(list[imNo][0])
# vecs = list[imNo][1].astype('uint16')

# p = pose_center_point_for_rotation(vecs)
# print p
# im = rotate_image(im, p, 20)
# rot = rotate_joints(vecs, p, 20)
# bbox = enlarged_pose_bbox(rot)
# im = cv2.rectangle(im, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), thickness=2,  lineType=8, shift=0)
# plot_image_skeleton_for_testing(im, rot)



# create_test_and_train_set()

train_net()
