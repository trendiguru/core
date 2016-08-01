import cv2
import numpy as np
import os
import sys
import scipy.io as sio
import h5py

# from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.layers import merge, Input, Dense, Dropout, Activation, Flatten, Convolution2D, Lambda, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2, activity_l1l2
from keras.utils import np_utils
from sklearn.cluster import MiniBatchKMeans#, k_means
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def texture_images_maker_for_featuring(images, masks, patch_size0=(32, 32)):
    # masks and images must be same length and correlated
    # the mask needs to be a convex nonzero blob in which the texture is to be evaluated
    patches_fot_inspection = []
    kernel = np.ones((5, 5), dtype='uint8')
    for i in range(len(images)):

        # 1 : erode mask
        erosion = cv2.erode(masks[i].copy(), kernel, iterations=1)
        # 2 : distance transform, than find max value
        dist_transform = cv2.distanceTransform(erosion, cv2.DIST_L2, 5)
        center = np.argmax(dist_transform)
        # 3 : at max value, expand to the sized square, and check if all is unmasked.
        patch_size = patch_size0
        patch = masks[i][center[0]-patch_size[0]/2:center[0]+patch_size[0]/2,
                center[1]-patch_size[1]/2:center[1]-patch_size[1]/2]
        # 4 : else 3, move square to make patch unmasked
        if patch.count_nonzero < patch_size[0] * patch_size[1]:
            patch_size = np.array(patch_size, dtype='int')
            while patch.count_nonzero < patch_size[0] * patch_size[1]:
                patch_size -= 1
                patch = masks[i][center[0]-patch_size[0]/2:center[0]+patch_size[0]/2,
                                center[1]-patch_size[1]/2:center[1]-patch_size[1]/2]

        patch_fot_inspection = images[i][[center[0]-patch_size[0]/2:center[0]+patch_size[0]/2,
                               center[1]-patch_size[1]/2:center[1]-patch_size[1]/2]]
        if patch_size[0] < patch_size0[0] or patch_size[1] < patch_size0[1]:
            patch_fot_inspection = cv2.resize(patch_fot_inspection, patch_size0)

        patch_fot_inspection = cv2.cvtColor(patch_fot_inspection, cv2.COLOR_BGR2GRAY)
        patch_fot_inspection = cv2.equalizeHist(patch_fot_inspection)
        patch_fot_inspection = patch_fot_inspection.reshape((patch_size0[0], patch_size0[1], 1))
        patches_fot_inspection.append(patch_fot_inspection)

    patches_fot_inspection = np.array(patches_fot_inspection, dtype='uint8')
    return patches_fot_inspection


def net(input_shape=(1, 32, 32)):
    input_img = Input(input_shape)
    # convolutional layers: (N_kernels, h_kernel, W_kernel)
    conv = BatchNormalization()(input_img)
    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Convolution2D(32, 2, 2, activation='relu', border_mode='valid')(conv)
    fc = Flatten()(conv)

    # fully connected layers: (N_newrons)
    # fc = Dense(16, activation='relu')(fc)
    # # fc = Dropout(0.25)(fc)
    fc = Dense(5, activation='softmax')(fc)
    model = Model(input=input_img, output=fc)
    optimizer_method = 'adam'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method, metrics=['accuracy'])
    model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'texture_model_weights.hdf5'))

    return model


def texture_feature_neural_net(model, texture_images):

    size_batch = len(texture_images)

    texture_images = np.array(texture_images, dtype='float32') / 255
    texture_images = np.transpose(texture_images, (0, 3, 1, 2))
    result = model.predict_proba(texture_images, batch_size=size_batch)

    return result


def texture_featurizer(model, image, mask):
    texture_images = texture_images_maker_for_featuring(image, mask)
    return texture_feature_neural_net(model, texture_images)


def texture_distance(texture1_vec, texture2_vec, weights_vec):
    # alll inputs must be of similar shape numpy float32 arrays
    dC12 = cv2.compareHist(texture1_vec * weights_vec,
                           texture2_vec * weights_vec, cv2.HISTCMP_CHISQR)
    return dC12