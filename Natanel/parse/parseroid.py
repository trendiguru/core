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
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation, Flatten, Reshape, Permute
# from keras.layers.advanced_activations import LeakyRelu, Prelu
from keras.layers.noise import GaussianDropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D#, Convolution1D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2, activity_l1l2
import time


def resize_images(images_list, output_shape=(128, 128)):
    '''
    :param images_list:
    :param output_shape:
    :return:
    '''

    border_type = cv2.BORDER_REPLICATE
    output_images_list = []
    for image0 in images_list:
        # no change besides resizing (to output size):
        # getting image size:
        output_images_size = image0.shape[:2]
        #shrinking oversized images:
        if image0.shape[0] > output_shape[0] or image0.shape[1] > output_shape[1]:
            if image0.shape[0] > image0.shape[1]:
                scale = 1.0*output_shape[0]/image0.shape[0]
            else:
                scale = 1.0*output_shape[1]/image0.shape[1]

            image0 = cv2.resize(np.array(image0, dtype='uint8'), (int(image0.shape[1] * scale), int(image0.shape[0] * scale)))
            output_images_size = image0.shape[:2]

        h, w = output_images_size
        w_shift = output_shape[1] - w
        h_shift = output_shape[0] - h
        image000 = cv2.copyMakeBorder(image0, h_shift/2, h_shift/2, w_shift/2, w_shift/2, border_type)

        # setting all to output_shape:
        output_images_list.append(cv2.resize(image000, output_shape).astype('uint8'))

    return output_images_list, scale, h_shift, w_shift


def masks_to_png_converter(masks):
    masks_shape = masks.shape
    yy = np.zeros(masks_shape[:2])
    for mask_id in range(masks_shape[2]):
        yy[masks[:, :, mask_id] > 0] = mask_id

    return yy.astype('uint8')


def convert_png_to_class_parse(png):

    # png = cv2.imread(png_file_path, 0)
    png_shape = list(png.shape) + [23]
    yy = np.zeros(png_shape, dtype='uint8')
    for mask_id in range(png_shape[2]):
        yy[:, :, mask_id][png[:, :]==mask_id] = 1 #mask_id
        # cv2.imshow('p', yy[:, :, mask_id]*255)
        # cv2.waitKey(0)
    return yy #np.array(yy, dtype=bool)


def TG23_parser():
    '''

    :return:
    '''

    output_shape = (128, 128)
    model_description = 'parsing_model_weights_128'
    fully_connected_layer_size = 2 ** 12
    max_shape_images = (3,) + output_shape
    max_shape_masks = (23,) + output_shape
    print max_shape_images, max_shape_masks

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

    ######################################################
    input_img = Input(shape=max_shape_images)
    conv = BatchNormalization()(input_img)

    conv = Convolution2D(64, 5, 5, border_mode='same')(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)

    conv = Convolution2D(128, 3, 3, border_mode='same')(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)

    conv = Convolution2D(256, 3, 3, border_mode='same')(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)

    conv = Convolution2D(512, 3, 3, border_mode='same')(conv)
    conv = Activation(act)(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)

    conv = Convolution2D(128, 1, 1, border_mode='same')(conv)
    conv = Activation(act)(conv)

    fc = Flatten()(conv)
    fc = Dense(4096, activation=act)(fc)
    fc = Dense(4096, activation=act)(fc)
    fc = Dense(2*16384, activation=act)(fc)
    conv = Reshape((2, 128, 128))(fc)

    conv = Convolution2D(32, 3, 3, border_mode='same')(conv)
    conv = Activation(act)(conv)

    conv = Convolution2D(128, 3, 3, border_mode='same')(conv)
    conv = Activation(act)(conv)

    conv = Convolution2D(64, 1, 1, border_mode='same')(conv)
    conv = Activation(act)(conv)

    conv = Convolution2D(23, 1, 1, border_mode='same')(conv)
    conv = Reshape((23, 128 * 128))(conv)
    conv = Permute((2, 1))(conv)
    conv = Activation('softmax')(conv)

    model = Model(input=input_img, output=conv)
    model.summary()

    optimizer_method = 'adam'  # SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()#Adadelta()#
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method, metrics=['accuracy'])
    if os.path.isfile(model_description + '.hdf5'):
        print 'loading weights file: ' + os.path.join(model_description + '.hdf5')
        model.load_weights(model_description + '.hdf5')
        return model  # model is a neural net object
    else:
        print 'no model weights file (*.hdf5) was found... model was not built.'


def masking23(model, images_list):
    '''

    :param model:
    :param images_list:
    :return:
    '''

    size_batch = len(images_list)
    if size_batch == 0:
        print 'no face, so no collar detected.'
        return

    images_list = np.array(images_list, dtype='float32') / 255
    images_list = np.transpose(images_list, (0, 3, 1, 2))

    # proba = model.predict_proba(images_list, batch_size=size_batch)
    # classes = model.predict_classes(images_list, batch_size=size_batch)
    # result = []
    # for image in images_list:
    #     result.append(model(image))

    result = model.predict(images_list, batch_size=size_batch)
    # result = np.transpose(result, (0, 2, 3, 1))# changing from (channels, h, w) to (h, w, channels)
    result = np.reshape(result, (128, 128, 23))
    return result #, proba #, classes


def run_test(im, output_shape=(128, 128)):

    model = TG23_parser()
    images_list = [im]
    images_list, scale, h_shift, w_shift = resize_images(images_list, output_shape=output_shape)
    tic = time.clock()
    results = masking23(model, images_list)
    toc = time.clock() - tic

    print ' time elapsed for ', len(images_list), 'images NN run is: ', toc, ' seconds.'

    # resizing class mask:
    results = cv2.resize(results, (max(im.shape[:2]), max(im.shape[:2])))
    results = results[int(h_shift/scale)/2:results.shape[0] - int(h_shift/scale)/2,
              int(w_shift/scale)/2:results.shape[1] - int(w_shift/scale)/2, :]

    singelton = np.argmax(results, axis=2)

    # layers = np.vstack([np.hstack([results[:, :, 0], results[:, :,0], results[:, :, 1], results[:, :, 2],
    #                                       results[:, :, 3], results[:, :, 4], results[:, :, 5], results[:, :, 6]]),
    #                            np.hstack([results[:, :, 7], results[:, :, 8], results[:, :, 9], results[:, :, 10],
    #                                       results[:, :, 11], results[:, :, 12], results[:, :, 13], results[:, :, 14]]),
    #                            np.hstack([results[:, :, 15], results[:, :, 16], results[:, :, 17], results[:, :, 18],
    #                                       results[:, :, 19], results[:, :, 20], results[:, :, 21], results[:, :, 22]])])
    #
    # layers = (layers * 255).astype('uint8')
    # layers = np.dstack([layers, layers, layers])
    # layers[:128, :128, :] = im
    #
    # cv2.imshow('p', layers)
    # cv2.waitKey(0)

    return results, singelton.astype('uint8')


def display_mask(mask):

    # 'categories' differs from '_categories' by only 'sunglasses' vs 'sunglass' respectively...
    categories =['bk', 'T-shirt', 'bag', 'belt', 'blazer', 'blouse', 'coat', 'dress', 'face',
                  'hair', 'hat', 'jeans', 'legging', 'pants', 'scarf', 'shoes', 'shorts', 'skin',
                  'skirt', 'socks', 'stocking', 'sunglasses', 'sweater']

    # colors in BGR opencv space:
    colors = {'bk': (0, 0, 0),
              'T-shirt': (255, 50, 50),
              'bag': (50, 50, 50),
              'belt': (45, 110, 45),
              'blazer': (215, 160, 0),
              'blouse': (200, 0, 0),
              'coat': (160, 160, 160),
              'dress': (0, 0, 255),
              'face': (250, 210, 255),
              'hair': (0, 30, 75),
              'hat': (0, 150, 255),
              'jeans': (100, 30, 10),
              'legging': (50, 255, 255),
              'pants': (30, 160, 160),
              'scarf': (255, 80, 255),
              'shoes': (0, 255, 0),
              'shorts': (255, 255, 0),
              'skin': (180, 220, 255),
              'skirt': (120, 0, 255),
              'socks': (120, 255, 150),
              'stocking': (30, 200, 225),
              'sunglasses': (120, 0, 120),
              'sweater': (45, 130, 100)}

    Bmask = mask.copy()
    Gmask = mask.copy()
    Rmask = mask.copy()

    for mask_id in range(23):
        Bmask[Bmask == mask_id] = colors[categories[mask_id]][0]
        Gmask[Gmask == mask_id] = colors[categories[mask_id]][1]
        Rmask[Rmask == mask_id] = colors[categories[mask_id]][2]

    BGRmask = cv2.merge([Bmask, Gmask, Rmask])

    unq = np.unique(mask)
    unq = unq[unq>0]
    block0 = np.zeros((mask.shape[0], 100, 3))
    index1 = 0
    index2 = 1
    block = block0.copy()
    for cat in unq:
        index1 += 1
        if 1. * index1 % (mask.shape[0]/16 + 1) == 0:
            index1 = 1
            index2 += 1
            block = np.hstack([block, block0.copy()])
        text_block = np.ones((16, 100, 3)) * colors[categories[cat]]
        cv2.putText(text_block, categories[cat], (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), thickness=1)
        block[16*(index1-1):16*index1, 100*(index2-1):100*index2, :] = text_block

    BGRmask = np.hstack([BGRmask, block]).astype('uint8')

    return BGRmask


# num = 300
# im = cv2.imread('/home/nate/Desktop/cloths_parsing/TG_dataset/im_' + str(num) + '.jpg')
# png = cv2.imread('/home/nate/Desktop/cloths_parsing/TG_dataset/msk_' + str(num) + '.png', 0)
# BGRmask = display_mask(png)
# print BGRmask.shape
# results, singelton = run_test(im)
# net_mask = display_mask(singelton)
#
# cv2.imshow('p', np.hstack([im, BGRmask, net_mask]))
# cv2.waitKey(0)


im = cv2.imread('/home/nate/Desktop/8c30a2582a64434c87fe0c504e2c1640.jpg')
results, singelton = run_test(im)
net_mask = display_mask(singelton)
print im.shape, net_mask.shape
cv2.imshow('p', np.hstack([im, net_mask]))
cv2.waitKey(0)
