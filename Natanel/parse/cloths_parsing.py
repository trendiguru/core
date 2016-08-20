import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import sys
# import tables
import scipy
from scipy import ndimage
import scipy.io as sio
import json
import h5py
from keras import backend as K
# from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation, Flatten, Reshape, Lambda, Permute
# from keras.layers.advanced_activations import LeakyRelu, Prelu
from keras.layers.noise import GaussianDropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D#, Convolution1D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2, activity_l1l2
from keras import backend as K
import theano
from keras.utils import np_utils
# from scipy.cluster.vq import whiten

# import scipy as sp
# import argparse
# from __future__ import print_function

#########################################################################################
#one timers:
def output_masks_creator(path_to_HDF5):

    data = h5py.File(path_to_HDF5)
    keys = data.keys()
    category_label = np.array(data[keys[0]], dtype='int')-1
    # color_label = np.array(data[keys[1]], dtype='int')
    segmentation = np.array(data[keys[2]], dtype='int').T
    output_masks = np.zeros((segmentation.shape[0], segmentation.shape[1], 23), dtype='uint8')
    # TODO: indexwise optimizes loop...    output_masks[]
    for i in range(len(category_label)):
        p = (segmentation == i)
        output_masks[:, :, category_label[i]] = output_masks[:, :, category_label[i]] + p

    return output_masks

def ground_truth_masks_converter():
    current_directory_name = os.getcwd()
    path_to_images = current_directory_name + '/_data_images/'
    path_to_images_data = current_directory_name + '/image_data/'
    images = [f for f in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, f))]
    images_data = [f for f in os.listdir(path_to_images_data) if os.path.isfile(os.path.join(path_to_images_data, f))]

    # creating HDF5 for training:
    hdf5_dataset_file_name = 'cloths_parsing_dataset'
    with h5py.File(hdf5_dataset_file_name + '.hdf5', 'w') as f:
        for image_data in images_data:
            output_masks = output_masks_creator(path_to_images_data + image_data)
            f.create_dataset(image_data[:-5], data=output_masks)
            print image_data[:-5] + '   done!'

# converting yamaguchi's dataset from 56 classes to 21 classes:
def slim_down_class_matrixes_from_56_to_21(_56_maskoid):

    # _56_maskoid: a cv2.imread() result (3 identical channles)
    _56_maskoid = _56_maskoid[:, :, 0] - 1
    output_masks = np.zeros((_56_maskoid.shape[0], _56_maskoid.shape[1], 23), dtype='uint8')

    ## base lineclasses:
    _categories =['bk', 'T-shirt', 'bag', 'belt', 'blazer', 'blouse', 'coat', 'dress', 'face',
                  'hair', 'hat', 'jeans', 'legging', 'pants', 'scarf', 'shoe', 'shorts', 'skin',
                  'skirt', 'socks', 'stocking', 'sunglass', 'sweater']

    ## fashionista classes:
    fashionista_categories = ['null','tights','shorts','blazer','t-shirt','bag','shoes','coat','skirt','purse',
                            'boots','blouse','jacket','bra','dress','pants','sweater','shirt','jeans','leggings',
                            'scarf','hat','top','cardigan','accessories','vest','sunglasses','belt','socks','glasses',
                            'intimate','stockings','necklace','cape','jumper','sweatshirt','suit','bracelet','heels','wedges',
                            'ring','flats','tie','romper','sandals','earrings','gloves','sneakers','clogs','watch',
                            'pumps','wallet','bodysuit','loafers','hair','skin','face']

    conversion_dictionary_strings = {'bk': ['null'],
                                    'T-shirt': ['t-shirt', 'shirt'],
                                    'bag': ['bag', 'purse', 'accessories', 'ring', 'necklace', 'bracelet', 'wallet', 'tie', 'earrings', 'gloves', 'watch'],
                                    'belt': ['belt'],
                                    'blazer': ['blazer', 'jacket', 'vest'],
                                    'blouse': ['blouse', 'bra', 'top', 'sweatshirt'],
                                    'coat': ['coat', 'cape'],
                                    'dress': ['dress', 'suit', 'bodysuit', 'romper'],
                                    'face': ['face'],
                                    'hair': ['hair'],
                                    'hat': ['hat'],
                                    'jeans': ['jeans'],
                                    'legging': ['tights', 'leggings'],
                                    'pants': ['pants'],
                                    'scarf': ['scarf'],
                                    'shoe': ['shoes', 'boots', 'heels', 'wedges', 'pumps', 'loafers', 'flats', 'sandals', 'sneakers', 'clogs'],
                                    'shorts': ['shorts'],
                                    'skin': ['skin'],
                                    'skirt': ['skirt'],
                                    'socks': ['socks'],
                                    'stocking': ['intimate', 'stockings'],
                                    'sunglass': ['sunglasses', 'glasses'],
                                    'sweater': ['sweater', 'cardigan', 'jumper']}

    for i in range(output_masks.shape[2]):
        list_56_class_match = conversion_dictionary_strings[_categories[i]]
        # print '\n category :: ' + _categories[i] + ':'
        for _56_class_match in list_56_class_match:
            # print '   # ' + _56_class_match
            output_masks[:, :, i][_56_maskoid == fashionista_categories.index(_56_class_match)] = 1

    # cv2.imshow('p', np.hstack([cv2.imread('fashionista_images_and_masks/1_photo.jpg'), (cv2.imread('fashionista_images_and_masks/1_mask.png')-1)*8, cv2.merge([p, p, p])*15]))
    # cv2.waitKey(0)

    _21_maskoid = output_masks.astype('uint8')
    return _21_maskoid

def fashionista_ground_truth_masks_converter():
    current_directory_name = os.getcwd()
    path_to_images = current_directory_name + '/fashionista_images_and_masks/'
    path_to_images_data = current_directory_name + '/fashionista_images_and_masks/'
    images = [f for f in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, f)) and f[-3:]=='jpg']
    # images_data = [f for f in os.listdir(path_to_images_data) if os.path.isfile(os.path.join(path_to_images_data, f)) and f[-3:]=='png']
    print images
    # creating HDF5 for training:
    hdf5_dataset_file_name = 'fashionista_cloths_parsing_dataset'
    with h5py.File(hdf5_dataset_file_name + '.hdf5', 'w') as f:
        for image_name in images:
            _56_maskoid = cv2.imread(path_to_images_data + image_name.split('_')[0] + '_mask.png')
            output_masks = slim_down_class_matrixes_from_56_to_21(_56_maskoid)
            f.create_dataset(image_name, data=output_masks)
            print image_name + '   done!'

#########################################################################################

def load_XandY():

    _data_images_path = '_data_images/'
    fashionista_data_images_path = 'fashionista_images_and_masks/'
    _data = h5py.File('cloths_parsing_dataset.hdf5')
    fashionista_data = h5py.File('fashionista_cloths_parsing_dataset.hdf5')
    _images_file_names = _data.keys()
    fashionista_images_file_names = fashionista_data.keys()
    print fashionista_images_file_names
    X = []
    Y = []

    # first aquier _data and then concatenate the fashionista_data:
    for image_name in _images_file_names:
        X.append(cv2.imread(_data_images_path + image_name))
        Y.append(_data[image_name])
    # print 'dont forget to remove break and unlock fashionista!!! (lines 149-153)'
    for image_name in fashionista_images_file_names:
        X.append(cv2.imread(fashionista_data_images_path + image_name))
        Y.append(fashionista_data[image_name])

    return X, Y


def relevant_cuts_of_Xi_and_Yi(Xi, Yi, output_shape=(128, 128)):

    image0 = Xi#.astype('uint8')
    masks0 = Yi#.astype('uint8')

    # bbox for speed slicing:
    face_mask = masks0[:, :, 8] # index=8 is face blob mask
    face_x0, face_y0, face_dx, face_dy = cv2.boundingRect(face_mask.astype('uint8'))
    human_mask = (masks0[:, :, 0]-1)**2 # index=0 is background to human blob mask
    body_x0, body_y0, body_dx, body_dy = cv2.boundingRect(human_mask.astype('uint8'))
    Xi_list = []
    Yi_list = []
    margine_pixels = 20
    border_type = cv2.BORDER_REPLICATE
    y0 = body_y0-margine_pixels
    x0 = body_x0-margine_pixels
    if y0 < 0:
        y0 = 0
    if x0 < 0:
        x0 = 0
    image1 = image0[y0:body_y0+body_dy+margine_pixels,
                    x0:body_x0+body_dx+margine_pixels, :]
    masks1 = masks0[y0:body_y0+body_dy+margine_pixels,
                    x0:body_x0+body_dx+margine_pixels, :]


    # first - non resized square cuts:
    ##################################
    DX_smallest = min(image0.shape[:2])
    scale_XY = 1.0*DX_smallest/max(image0.shape[:2])
    if max(image0.shape[:2]) >= 1.5*max(output_shape):
        if scale_XY <= 2./3:
            if image0.shape[0] >= image0.shape[1]:
                # up:
                Xi_list.append(image0[:DX_smallest, :, :])
                Yi_list.append(masks0[:DX_smallest, :, :])
                # down:
                Xi_list.append(image0[-DX_smallest:, :, :])
                Yi_list.append(masks0[-DX_smallest:, :, :])
            if image0.shape[0] < image0.shape[1]:
                # left:
                Xi_list.append(image0[:, :DX_smallest, :])
                Yi_list.append(masks0[:, :DX_smallest, :])
                #right:
                Xi_list.append(image0[:, -DX_smallest:, :])
                Yi_list.append(masks0[:, -DX_smallest:, :])

            if 2./3 > scale_XY >= 1. / 3:
                if image0.shape[0] >= image0.shape[1]:
                    # up:
                    Xi_list.append(image0[:image0.shape[1], :, :])
                    Yi_list.append(masks0[:masks0.shape[1], :, :])
                    # midway:
                    Xi_list.append(image0[image0.shape[0]/2-image0.shape[1]/2:image0.shape[0]/2+image0.shape[1]/2, :, :])
                    Yi_list.append(masks0[masks0.shape[0]/2-masks0.shape[1]/2:masks0.shape[0]/2+masks0.shape[1]/2, :, :])
                    # down:
                    Xi_list.append(image0[-image0.shape[1]:, :, :])
                    Yi_list.append(masks0[-masks0.shape[1]:, :, :])
                if image0.shape[0] < image0.shape[1]:
                    # up:
                    Xi_list.append(image0[:, image0.shape[0], :])
                    Yi_list.append(masks0[:, masks0.shape[0], :])
                    # midway:
                    Xi_list.append(image0[:, image0.shape[1]/2-image0.shape[0]/2:image0.shape[1]/2+image0.shape[0]/2, :])
                    Yi_list.append(masks0[:, masks0.shape[1]/2-masks0.shape[0]/2:masks0.shape[1]/2+masks0.shape[0]/2, :])
                    # down:
                    Xi_list.append(image0[:, -image0.shape[0]:, :])
                    Yi_list.append(masks0[:, -masks0.shape[0]:, :])

            # TODO: ...
            # if scale_XY < 1. / 3:
            #     if image0.shape[0] > image0.shape[1]:
            #     if image0.shape[0] < image0.shape[1]:

        else:
            ## two steps:
            if image0.shape[0] >= image0.shape[1]:
                # upper left corner:
                Xi_list.append(image0[:2*image0.shape[0]/3, :2*image0.shape[0]/3, :])
                Yi_list.append(masks0[:2*masks0.shape[0]/3, :2*masks0.shape[0]/3, :])
                # upper right corner:
                Xi_list.append(image0[:2*image0.shape[0]/3, -2*image0.shape[0]/3:, :])
                Yi_list.append(masks0[:2*masks0.shape[0]/3, -2*masks0.shape[0]/3:, :])
                # middle:
                Xi_list.append(image0[image0.shape[0]/6:5*image0.shape[0]/6, image0.shape[0]/6 - (image0.shape[0]-image0.shape[1])/2:5*image0.shape[0]/6 - (image0.shape[0]-image0.shape[1]), :])
                Yi_list.append(masks0[masks0.shape[0]/6:5*masks0.shape[0]/6, masks0.shape[0]/6 - (masks0.shape[0]-masks0.shape[1])/2:5*masks0.shape[0]/6 - (masks0.shape[0]-masks0.shape[1]), :])
                # lower left corner:
                Xi_list.append(image0[-2*image0.shape[0]/3:, :2*image0.shape[0]/3, :])
                Yi_list.append(masks0[-2*masks0.shape[0]/3:, :2*masks0.shape[0]/3, :])
                # lower right corner:
                Xi_list.append(image0[-2 * image0.shape[0]/3:, -2 * image0.shape[0]/3:, :])
                Yi_list.append(masks0[-2 * masks0.shape[0]/3:, -2 * masks0.shape[0]/3:, :])

            if image0.shape[0] < image0.shape[1]:
                # upper left corner:
                Xi_list.append(image0[:2*image0.shape[1]/3, :2*image0.shape[1]/3, :])
                Yi_list.append(masks0[:2*masks0.shape[1]/3, :2*masks0.shape[1]/3, :])
                # upper right corner:
                Xi_list.append(image0[:2*image0.shape[1]/3, -2*image0.shape[1]/3:, :])
                Yi_list.append(masks0[:2*masks0.shape[1]/3, -2*masks0.shape[1]/3:, :])
                # middle:
                Xi_list.append(image0[image0.shape[1]/6:5*image0.shape[1]/6, image0.shape[1]/6 - (image0.shape[1]-image0.shape[0])/2:5*image0.shape[1]/6 - (image0.shape[1]-image0.shape[0]), :])
                Yi_list.append(masks0[masks0.shape[1]/6:5*masks0.shape[1]/6, masks0.shape[1]/6 - (masks0.shape[1]-masks0.shape[0])/2:5*masks0.shape[1]/6 - (masks0.shape[1]-masks0.shape[0]), :])
                # lower left corner:
                Xi_list.append(image0[-2*image0.shape[1]/3:, :2*image0.shape[1]/3, :])
                Yi_list.append(masks0[-2*masks0.shape[1]/3:, :2*masks0.shape[1]/3, :])
                # lower right corner:
                Xi_list.append(image0[-2 * image0.shape[1]/3:, -2 * image0.shape[1]/3:, :])
                Yi_list.append(masks0[-2 * masks0.shape[1]/3:, -2 * masks0.shape[1]/3:, :])

        # smallest relevant square (non resized):
        # average of body and face bounding boxes size....
        face_dz = np.sqrt(face_dx*face_dy)#(face_dx**2 + face_dy**2)
        body_dz = np.sqrt(body_dx*body_dy)#(body_dx ** 2 + body_dy ** 2)
        D = int((face_dz+body_dz)/2)
        if D < min(output_shape):
            D = min(output_shape)
        Dy_partition = np.ceil(1.0 * image1.shape[0]/D)
        Dx_partition = np.ceil(1.0 * image1.shape[1]/D)
        X_coordinates = range(0, image1.shape[1]+1, int(image1.shape[1]/Dx_partition))
        Y_coordinates = range(0, image1.shape[0]+1, int(image1.shape[0]/Dy_partition))
        for y in Y_coordinates[:-1]:
            for x in X_coordinates[:-1]:
                if x < X_coordinates[-2] and y < Y_coordinates[-2]:
                    Xi_list.append(image1[y:y+D, x:x+D, :])
                    Yi_list.append(masks1[y:y+D, x:x+D, :])
                if x == X_coordinates[-2] and y < Y_coordinates[-2]:
                    Xi_list.append(image1[y:y+D, -D:, :])
                    Yi_list.append(masks1[y:y+D, -D:, :])
                if y == Y_coordinates[-2] and x < X_coordinates[-2]:
                    Xi_list.append(image1[-D:, x:x+D, :])
                    Yi_list.append(masks1[-D:, x:x+D, :])
                if x == X_coordinates[-2] and y == Y_coordinates[-2]:
                    Xi_list.append(image1[-D:, -D:, :])
                    Yi_list.append(masks1[-D:, -D:, :])
        # TODO: ##############################################
        # TODO: ##############################################
        # TODO: ##############################################


    # now - resized cuts from face side entire image:
    ################################################
    ## images:
    im = []
    im.append(image0[:, face_x0:, :]) # cut left side of face - including face
    im.append(image0[:, :face_x0+face_dx, :]) # cut right side of face - including face
    im.append(image0[:, face_x0+face_dx:, :]) # cut left side of face - not including face
    im.append(image0[:, :face_x0, :]) # cut right side of face - not including face
    im.append(image0[face_y0+face_dy:, face_x0:, :]) # cut left side of face - including face - below face
    im.append(image0[face_y0+face_dy:, :face_x0+face_dx, :]) # cut right side of face - including face - below face
    im.append(image0[face_y0+face_dy:, face_x0+face_dx:, :]) # cut left side of face - not including face - below face
    im.append(image0[face_y0+face_dy:, :face_x0, :]) # cut right side of face - not including face - below face
    ## masks:
    mask = []
    mask.append(masks0[:, face_x0:, :]) # cut left side of face - including face
    mask.append(masks0[:, :face_x0+face_dx, :]) # cut right side of face - including face
    mask.append(masks0[:, face_x0+face_dx:, :]) # cut left side of face - not including face
    mask.append(masks0[:, :face_x0, :]) # cut right side of face - not including face
    mask.append(masks0[face_y0+face_dy:, face_x0:, :]) # cut left side of face - including face - below face
    mask.append(masks0[face_y0+face_dy:, :face_x0+face_dx, :]) # cut right side of face - including face - below face
    mask.append(masks0[face_y0+face_dy:, face_x0+face_dx:, :]) # cut left side of face - not including face - below face
    mask.append(masks0[face_y0+face_dy:, :face_x0, :]) # cut right side of face - not including face - below face

    for i in range(8):
        size = im[i].shape[:2]
        if size[0] > 0 and size[1] > 0:
            dshift = abs(size[0] - size[1])
            left_background_brim_much = np.count_nonzero(mask[i][:, 0, 0]>0)
            right_background_brim_much = np.count_nonzero(mask[i][:, -1, 0]>0)
            top_background_brim_much = np.count_nonzero(mask[i][0, :, 0]>0)
            bottom_background_brim_much = np.count_nonzero(mask[i][-1, :, 0]>0)

            if 1./3 <= 1.*size[0]/size[1] <= 3:
                if size[0] > size[1]: # vertical longer
                    if left_background_brim_much >= right_background_brim_much:
                        Xi_list.append(cv2.copyMakeBorder(im[i], 0, 0, dshift, 0, border_type))
                        yy = np.zeros((max(size), max(size), masks0.shape[-1]))
                        yy[:, -size[1]:, :] = mask[i]
                        yy[:, :-size[1], 0] = 1
                        Yi_list.append(yy.astype('uint8'))
                    else:
                        Xi_list.append(cv2.copyMakeBorder(im[i], 0, 0, 0, dshift, border_type))
                        yy = np.zeros((max(size), max(size), masks0.shape[-1]))
                        yy[:, :size[1], :] = mask[i]
                        yy[:, size[1]:, 0] = 1
                        Yi_list.append(yy.astype('uint8'))

                elif size[0] < size[1]: # horizontal longer
                    if top_background_brim_much >= bottom_background_brim_much:
                        Xi_list.append(cv2.copyMakeBorder(im[i], dshift, 0, 0, 0, border_type))
                        yy = np.zeros((max(size), max(size), masks0.shape[-1]))
                        yy[-size[1]:, :, :] = mask[i]
                        yy[:-size[1], :, 0] = 1
                        Yi_list.append(yy.astype('uint8'))
                    else:
                        Xi_list.append(cv2.copyMakeBorder(im[i], 0, dshift, 0, 0, border_type))
                        yy = np.zeros((max(size), max(size), masks0.shape[-1]))
                        yy[:size[0], :, :] = mask[i]
                        yy[size[0]:, :, 0] = 1
                        Yi_list.append(yy.astype('uint8'))

                elif size[0] == size[1]:
                    Xi_list.append(im[i])
                    yy = mask[i]
                    Yi_list.append(yy.astype('uint8'))



    # no change besides resizing (to output size):
    # getting image size:
    output_images_size = image0.shape[:2]
    #shrinking oversized images:
    if image0.shape[0] > output_shape[0] or image0.shape[1] > output_shape[1]:
        if image0.shape[0] > image0.shape[1]:
            # output_images_size = (image.shape[1]*max_shape[0]/image.shape[0], max_shape[0])
            scale = 1.0*output_shape[0]/image0.shape[0]
        else:
            # output_images_size = (max_shape[1], image.shape[0]*max_shape[1]/image.shape[1])
            scale = 1.0*output_shape[1]/image0.shape[1]

        image0 = cv2.resize(np.array(image0, dtype='uint8'), (int(image0.shape[1] * scale), int(image0.shape[0] * scale)))
        masks0 = cv2.resize(np.array(masks0, dtype='uint8'), (int(masks0.shape[1] * scale), int(masks0.shape[0] * scale)))
        output_images_size = image0.shape[:2]

    h, w = output_images_size
    w_shift = output_shape[1] - w
    h_shift = output_shape[0] - h
    image000 = cv2.copyMakeBorder(image0, h_shift/2, h_shift/2, w_shift/2, w_shift/2, border_type)
    masks000 = cv2.copyMakeBorder(masks0, h_shift/2, h_shift/2, w_shift/2, w_shift/2, border_type)
    Xi_list.append(image000)
    Yi_list.append(masks000)

    # setting all cuts to output_shape:
    for i in range(len(Xi_list)):
        Xi_list[i] = cv2.resize(Xi_list[i], output_shape).astype('uint8')

    for i in range(len(Yi_list)):
        Yi_list[i] = cv2.resize(Yi_list[i], output_shape).astype('uint8')

    # for i in range(len(Xi_list[-8:])):
    #     cv2.imshow('x', Xi_list[i])
    #     cv2.waitKey(0)
    #     cv2.imshow('x', Yi_list[i][:, :, 0]*255)
    #     cv2.waitKey(0)

    return Xi_list, Yi_list


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


def flip_and_rotate(image, masks):

    image_list = []
    masks_list = []
    for degree in [0]:#, 90, 180, 270]:
        # images:
        im = ndimage.rotate(image.copy(), degree)
        image_list.append(im)
        fim = np.fliplr(im)
        image_list.append(fim)
        # masks
        msk = ndimage.rotate(masks.copy(), degree)
        masks_list.append(msk)
        fmsk = np.fliplr(msk)
        masks_list.append(fmsk)

    return image_list, masks_list


def variation_list(image, masks):

    Xi_list = []
    Yi_list = []
    X, Y = relevant_cuts_of_Xi_and_Yi(image, masks)
    for i in range(len(X)):
        XX, YY = flip_and_rotate(X[i], Y[i])
        for j in range(len(XX)):
            XXX = color_delta(XX[j])
            for k in range(len(XXX)):
                Xi_list.append(XXX[k])
                Yi_list.append(YY[j])

    return Xi_list, Yi_list


def masks_to_png_converter(masks):

    masks_shape = masks.shape
    yy = np.zeros(masks_shape[:2])
    for mask_id in range(masks_shape[2]):
        yy[masks[:, :, mask_id]>0] = mask_id

    return yy.astype('uint8')


def create_dataset(path_to_dataset_folder):

    X, Y = load_XandY()
    index = 0

    current_directory_name = os.getcwd()
    directory_path = current_directory_name + '/' + path_to_dataset_folder
    if not os.path.exists(path_to_dataset_folder):
        os.mkdir(path_to_dataset_folder)

    # # save data as HDF5:
    # with h5py.File(path_to_dataset_folder + '.hdf5', 'w') as f:
    #     for i in range(len(X[:1])):
    #         XX, YY = variation_list(X[i], Y[i])
    #         for j in range(len(XX)):
    #             f.create_dataset('im_' + str(index), data=np.array(XX[j], dtype='uint8'))
    #             f.create_dataset('msk_' + str(index), data=np.array(YY[j], dtype='uint8'))
    #             index += 1
    #         print i, ' done!'

    # save data as images

    # for learning:
    amount_of_images_for_post_testing = 200
    for i in range(len(X[:-amount_of_images_for_post_testing])):
        XX, YY = variation_list(X[i], Y[i])
        for j in range(len(XX)):
            cv2.imwrite(path_to_dataset_folder + '/im_' + str(index) + '.jpg', XX[j])
            cv2.imwrite(path_to_dataset_folder + '/msk_' + str(index) + '.png',  masks_to_png_converter(YY[j]))
            index += 1
        print i, 'for training done!'

    # for post testing:
    if not os.path.exists(path_to_dataset_folder + '/for_post_testing'):
        os.mkdir(path_to_dataset_folder + '/for_post_testing')
    for i in range(len(X[-amount_of_images_for_post_testing:])):
        XX, YY = variation_list(X[i], Y[i])
        for j in range(len(XX)):
            cv2.imwrite(path_to_dataset_folder + '/for_post_testing/test_im_' + str(index) + '.jpg', XX[j])
            cv2.imwrite(path_to_dataset_folder + '/for_post_testing/test_msk_' + str(index) + '.png', masks_to_png_converter(YY[j]))
            index += 1
        print i, 'for testing done!'


def convert_png_to_class_parse(png_file_path):

    png = cv2.imread(png_file_path, 0)
    png_shape = list(png.shape) + [23]
    yy = np.zeros(png_shape, dtype='uint8')
    for mask_id in range(png_shape[2]):
        yy[:, :, mask_id][png[:, :]==mask_id] = 1 #mask_id
        # cv2.imshow('p', yy[:, :, mask_id]*255)
        # cv2.waitKey(0)
    return yy #np.array(yy, dtype=bool)


def load_data(portion):
    '''
    :return:
    '''

    dataset_directory_name = 'dataset'
    current_directory_name = os.getcwd()
    directory_path = current_directory_name + '/' + dataset_directory_name
    only_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f[:3]=='msk']
    # only_files = only_files[only_files[:][:3] == 'msk']
    print 'length of file dataset is: ' + str(len(only_files))
    images = []
    masks = []

    only_files = only_files[:50000]
    p0 = int(portion[0]*len(only_files))
    p1 = int(portion[1]*len(only_files))

    for file_name in only_files[p0: p1]:
        name_split = file_name.split('_')
        designation = name_split[0]
        if designation == 'msk':
            mask_No = int(name_split[1].split('.')[0])
            # print mask_No
            # mask = cv2.imread(dataset_directory_name + '/' + file_name, 0)
            # mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
            mask = convert_png_to_class_parse(dataset_directory_name + '/' + file_name)
            # #
            # for y in range(mask.shape[0]):
            #     for x in range(mask.shape[1]):
            #         if np.count_nonzero(mask[y, x, :]) == 0:
            #             print y, x, ' => no class here!'

            ###
            mask = np.reshape(mask, (16384, 23)).T
            ###

            # masks.append([mask])
            masks.append(mask)
        # elif designation == 'im':
            image_No = int(name_split[1].split('.')[0])
            image = cv2.imread(dataset_directory_name + '/im_' + str(mask_No) + '.jpg', 1)
            images.append(image)

    images = np.array(images, dtype='float32') / 255
    masks = np.array(masks, dtype='float32') #/ 23

    print 'dataset size is: ' + str(sys.getsizeof(images)/1000000 + sys.getsizeof(masks)/1000000) + ' MB'
    return images, masks


def mini_batch(portion, testing_amount):

    images, masks = load_data(portion)
    # reshaping for requiered input shape:
    images = np.transpose(images, (0, 3, 1, 2))
    masks = np.transpose(masks, (0, 2, 1))
    # print images.shape, masks.shape
    # print max_shape_images, max_shape_masks
    data_length = len(masks)

    X_train = images[:int(1 - testing_amount * data_length)]
    Y_train = masks[:int(1 - testing_amount * data_length)]
    X_test = images[int(1 - testing_amount * data_length):]
    Y_test = masks[int(1 - testing_amount * data_length):]

    print Y_train.shape, Y_test.shape
    # print X_train.shape
    #
    # print Y_test.shape
    # print X_test.shape

    return (X_train, Y_train), (X_test, Y_test)


def depth_softmax(matrix):
    # sigmoid = lambda x: 1 / (1 + K.exp(-x))
    # sigmoided_matrix = sigmoid(matrix)
    # softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)

    # print softmax_matrix.shape
    # dmat0 = K.sum(matrix, axis=1) # 1
    # dmat = theano.tensor.stack([dmat0, dmat0, dmat0, dmat0, dmat0, dmat0, dmat0,
    #                              dmat0, dmat0, dmat0, dmat0, dmat0, dmat0, dmat0,
    #                              dmat0, dmat0, dmat0, dmat0, dmat0, dmat0, dmat0,
    #                              dmat0, dmat0], axis=1)  # 23

    softmax_matrix = matrix/matrix.norm(1, axis=1)#matrix/matrix.norm(1, axis=1)#.reshape((matrix.shape[0], 1))#theano.tensor.elemwise.Elemwise(theano.tensor.scalar.ScalarOp.true_div)(dmat, dmat0)#matrix / K.sum(matrix, axis=1)

    return softmax_matrix


def train_net():
    '''
    :return:
    '''

    model_description = 'parsing_model_weights'

    size_batch = 8#images[0].shape[-1]
    print size_batch

    dataset_Nof_steps = 10
    epoches_number = 1000000
    overwrite_weights = True
    testing_amount = 0.01
    # -----------------
    # Net META parameters:
    # main_Nkernels_down = 32
    # main_Nkernels_up = 64
    # funnel_down_layers_n = 5
    # funnel_up_layers_n = 3
    # fc_layers_n = 2
    Nmain_fc_neurons = 2 ** 12
    act = 'relu'
    pre_up_x_y = 16
    # -----------------
    (X_train, Y_train), (X_test, Y_test) = mini_batch((0.0, 0.001), 0.1)#testing_amount)
    max_shape_images = X_train[0].shape
    max_shape_masks = Y_train[0].shape
    print max_shape_images, max_shape_masks

######################################################
# funnel down net:
    # W_regularizer=l1l2(l1=0.0001, l2=0.0001), b_regularizer=None, activity_regularizer=activity_l1l2(l1=0.0001, l2=0.0001)
    input_img = Input(shape=max_shape_images)
    conv = BatchNormalization()(input_img)

    conv = Convolution2D(64, 5, 5, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # conv = Dropout(0.0)(conv)    

    conv = Convolution2D(128, 3, 3, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # conv = Dropout(0.05)(conv)

    conv = Convolution2D(256, 3, 3, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # conv = Dropout(0.1)(conv)

    conv = Convolution2D(512, 3, 3, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # conv = Dropout(0.15)(conv)

    # conv = Convolution2D(128, 1, 1, border_mode='same')(conv)
    # # conv = BatchNormalization()(conv)
    # conv = Activation(act)(conv)
    # # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)

    # conv = Convolution2D(1024, 1, 1, border_mode='same')(conv)
    # # conv = BatchNormalization()(conv)
    # conv = Activation(act)(conv)
    # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)

    fc = Flatten()(conv)
    fc = Dense(2*4096, activation=act)(fc)
    # fc = Dense(4096, activation=act)(fc)
    # fc = Dense(4096, activation=act)(fc)
    # fc = Dense(2*4096, activation=act)(fc)
    fc = Dense(16384, activation=act)(fc)
    conv = Reshape((4, 128/2, 128/2))(fc)

    conv = Convolution2D(256, 9, 9, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)

    conv = UpSampling2D((2, 2))(conv)
    conv = Convolution2D(256, 3, 3, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)

    conv = Convolution2D(256, 2, 2, border_mode='same')(conv)
    # # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)

    conv = Convolution2D(64, 1, 1, border_mode='same')(conv)
    # conv = BatchNormalization()(conv)
    conv = Activation(act)(conv)

    conv = Convolution2D(23, 1, 1, border_mode='same')(conv)
    #    conv = Reshape((max_shape_masks[0], max_shape_masks[1]*max_shape_masks[2]))(conv)
    conv = Reshape((23, 128 * 128))(conv)
    conv = Permute((2, 1))(conv)
    conv = Activation('softmax')(conv)

    model = Model(input=input_img, output=conv)
    model.summary()


    optimizer_method = 'adam'#SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()#Adadelta()#
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method, metrics=['accuracy'])
    # binary_crossentropy
    # categorical_crossentropy

    ############################################################################################
    # if previus file exist:
    # if os.path.isfile(model_description + '.hdf5'):
    #     print 'loading weights file: ' + os.path.join(model_description + '.hdf5')
    #     model.load_weights(model_description + '.hdf5')
    ############################################################################################

    EarlyStopping(monitor='val_loss', patience=0, verbose=1) #monitor='val_acc'
    checkpointer = ModelCheckpoint(model_description + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True)


    # # this will do preprocessing and realtime data augmentation
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=False,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    # # compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied)
    for epoch_No in range(epoches_number):
        for step in range(dataset_Nof_steps):
            p0 = 1. * step / dataset_Nof_steps
            p1 = p0 + 1. / dataset_Nof_steps
            portion = (p0, p1)
            (X_train, Y_train), (X_validation, Y_validation) = mini_batch(portion, testing_amount)
            print 'step No: ', step + 1, '/', dataset_Nof_steps, '...  @ epoch No: ', epoch_No + 1
            model.fit(X_train, Y_train, batch_size=size_batch, nb_epoch=1, verbose=1, callbacks=[checkpointer],
                      validation_split=0.0, validation_data=(X_validation, Y_validation), shuffle=True,
                      class_weight=None, sample_weight=None)
            #
            # datagen.fit(X_train)
            # # fit the model on the batches generated by datagen.flow()
            # model.fit_generator(datagen.flow(X_train, Y_train, shuffle=True, batch_size=size_batch),
            #                     nb_epoch=1, verbose=1, validation_data=(X_validation, Y_validation),
            #                     callbacks=[checkpointer], class_weight=None, max_q_size=10, samples_per_epoch=len(X_validation))
            # model.train_on_batch(X_train, Y_train)
            # model.test_on_batch(X_test, Y_test)

    model.save_weights(model_description + '.hdf5', overwrite_weights)


# fashionista_ground_truth_masks_converter()

# create_dataset('TG_dataset')

train_net()
