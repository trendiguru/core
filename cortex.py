__author__ = 'Natanel Davidovits'

import os
import pickle
import numpy as np
import cv2
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam


# def flok2float(num_str):
#     if num_str[0] == '0':
#         num = float('0.' + num_str[1:])
#     else:
#         num = float(num_str)
#     return num

def build_sequential_net_from_weights_filename(weights_filename):
    '''
    :param weights_filename: string of weights file - type HDF5, or will trigger error.
    :return: Keras sequential net net_object object - neural net object
    '''

    '''
    key to build by weights_filename's string (up to *.hdf5):
    NOTE: net_object layers are sequentially listed from left to right!
    1. slicing the layers is by a '_'
    2. slicing each layer data is by a 'x'
    3. 'C2D' = Sequential.add(Convolution2D())
        3.1 number of kernels (int)
        3.2 length of kernel along dim 1(int)
        3.3 length of kernel along dim 2(int)
        3.4 border_mode -> 'valid' / 'same'
        3.5 input_shape -> 3x dimentions (int)
    4. 'MP' = max pooling layer
        4.1 pool length dim 1 (int)
        4.2 pool length dim 2 (int)
    5. 'AP' = average pooling layer
        5.1 pool length dim 1 (int)
        5.2 pool length dim 2 (int)
    4. 'F' = Sequential.add(Flatten())
    5. 'A' = Sequential.add(Activation(->)) : 'relu' / 'sigmoid' / 'hard_sigmoid / 'softmax' / 'tanh' / 'softplus' / 'linear'
    6. 'DO' = Sequential.add(Dropout(->)) : 0.0 <= value <= 1.0
    7. 'D' = Sequential.add(Dense(->)) : int value > 0 (fully connected 1D layer of size *value*)
    8. 'OP' = optimizer_method
        8.1 'adagrad' = Adagrad()
        8.2 'adadelta' = Adadelta()
        8.3 'rmsprop' = RMSprop()
        8.4 'adam' = Adam()
        8.5 'sgd' = SGD():
            8.5.1 val1 : lr=val1 (float)
            8.5.2 val2 : decay=val2 (float)
            8.5.3 val3 : momentum=val3 (float)
            8.5.4 val4 : nesterov=val4 (bool, i.e. 1 / 0)
    9. 'CO' = Sequential.compile()
        9.1 loss :
            9.1.1 'mse' = mean_squared_error
            9.1.2 'mae' = mean_absolute_error
            9.1.3 'mape' = mean_absolute_percentage_error
            9.1.4 'msle = mean_absolute_percentage_error
            9.1.5 'sh' = squared_hinge
            9.1.5 'h' = hinge
            9.1.6 'bc' = binary_crossentropy (Also known as logloss)
            9.1.7 'p' = poisson: mean of (predictions - targets * log(predictions))
            9.1.8 'cp' = cosine_proximity: the opposite (negative) of the mean cosine proximity between predictions
            9.1.9 'cc' = categorical_crossentropy : Also known as multiclass logloss. Note: using this objective requires that your labels are binary arrays of shape (nb_samples, nb_classes).
    '''

    # isolating the relevant string:
    net_composition = weights_filename[:-5]

    # isolating layers:
    layers = net_composition.split('X')

    # case assembly:
    net_object = Sequential()
    #
    print layers
    print len(layers)-1
    #
    f = 0
    for layer in layers:
        f += 1
        print f
        layer_data = layer.split('x')
        #
        print layer
        print layer_data
        print len(layer_data)
        #
        if layer_data[0] == 'C2D': # len(layer_data) should be 8
            if len(layer_data) > 5:
                net_object.add(Convolution2D(int(layer_data[1]), int(layer_data[2]),
                                         int(layer_data[3]), border_mode=layer_data[4],
                                         input_shape=(int(layer_data[5]), int(layer_data[6]), int(layer_data[7]))))
            else:
                net_object.add(Convolution2D(int(layer_data[1]), int(layer_data[2]),
                                             int(layer_data[3]), border_mode=layer_data[4]))

        elif layer_data[0] == 'MP': # len(layer_data) should be 3
            net_object.add(MaxPooling2D(pool_size=(int(layer_data[1]), int(layer_data[2]))))

        elif layer_data[0] == 'AP': # len(layer_data) should be 3
            net_object.add(AveragePooling2D(pool_size=(int(layer_data[1]), int(layer_data[2]))))

        elif layer_data[0] == 'F': # len(layer_data) should be 1
            net_object.add(Flatten())

        elif layer_data[0] == 'A': # len(layer_data) should be 2
            if layer_data[1] == 'softma':
                layer_data[1] = 'softmax'
            net_object.add(Activation(layer_data[1]))

        elif layer_data[0] == 'DO': # len(layer_data) should be 2
            net_object.add(Dropout(float(layer_data[1])))

        elif layer_data[0] == 'D': # len(layer_data) should be 2
            net_object.add(Dense(int(layer_data[1])))

        elif layer_data[0] == 'OP': # len(layer_data) should be 2 to 5
            optimizer_method = layer_data[1]
            if optimizer_method == 'adagrad': # len(layer_data) should be 2
                optimizer_method = Adagrad()

            elif optimizer_method == 'adadelta': # len(layer_data) should be 2
                optimizer_method = Adadelta()

            elif optimizer_method == 'rmsprop': # len(layer_data) should be 2
                optimizer_method = RMSprop()

            elif optimizer_method == 'adam': # len(layer_data) should be 2
                optimizer_method = Adam()

            elif optimizer_method == 'sgd': # len(layer_data) should be 5
                if  layer_data[5] == 0:
                    nesterov_val = False
                else:
                    nesterov_val = True
                optimizer_method = SGD(lr=float(layer_data[2]),
                                       decay=float(layer_data[3]),
                                       momentum=float(layer_data[4]),
                                       nesterov=nesterov_val)

            else:
                print 'Error: no optimizer identifier (adagrad / adadelta / rmsprop / adam / sgd)'

        elif layer_data[0] == 'CO': # len(layer_data) should be 2

            if layer_data[1] == 'mse':
                loss_method = 'mean_squared_error'

            elif layer_data[1] == 'mae':
                loss_method = 'mean_absolute_error'

            elif layer_data[1] == 'mape':
                loss_method = 'mean_absolute_percentage_error'

            elif layer_data[1] == 'msle':
                loss_method = 'mean_absolute_percentage_error'

            elif layer_data[1] == 'sh':
                loss_method = 'squared_hinge'

            elif layer_data[1] == 'h':
                loss_method = 'hinge'

            elif layer_data[1] == 'bc':
                loss_method = 'binary_crossentropy'

            elif layer_data[1] == 'p':
                loss_method = 'poisson'

            elif layer_data[1] == 'cp':
                loss_method = 'cosine_proximity'

            elif layer_data[1] == 'cc':
                loss_method = 'categorical_crossentropy'

            net_object.compile(loss=loss_method, optimizer=optimizer_method)
            if layer != layers[-1]:
                print 'Error: compilation of neural net is not issued at the end of *weights_filename* string!'
        else:
            print 'Error: no layer identifier (C2D / F / A / DO / OP / CO)'

    net_object.load_weights(weights_filename)
    return net_object

def collar_images_maker_for_testing(image, face_box):
    '''
    :param image: openCV image array of BGR channels
    :param face_box: bounding box of face (as detected from the image
    :return: collar_images - a vector of images from single image, ready evaluate

    '''
    collar_images = []
    a = 1.35 # scalar for increasing collar box in relation to face box (1==100%)
    max_angle = 10 # tilt angle of the image for diversification
    angle_offset = 9 # tilt angle of the image for diversification
    output_images_size = (32, 32) # pixels^2

    a = (a-1)/2
    # checking if the face (ancore) is present / detected:
    if len(face_box) == 0:
        return
    face = face_box
    row, col, dep = image.shape
    if row < (face[1]+2*face[3])*(1+a):
        return collar_images
    # no flip along vertical axis:
    collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
    flipped_collar_image_center_point = (col - face[0]+0.5*face[2], face[1]+1.5*face[3])
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        if len(image_of_rotated_collar) > 0:
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                (offsetted_face[1]+2.1*offsetted_face[3])*(1+a),
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
            collar_images.append(resized_image_of_collar)

    # flip along vertical axis:
    image = np.fliplr(image)
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        if len(image_of_rotated_collar) > 0:
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                (offsetted_face[1]+2.1*offsetted_face[3])*(1+a),
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
            collar_images.append(resized_image_of_collar)


    collar_images = np.array(collar_images)
    images_vector_shape = collar_images.shape
    collar_images = np.reshape(collar_images, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))
    return collar_images

def collar_classifier_net():
    '''
    :param weights_filename: string of weights file - type HDF5, or will trigger error.
    :return ccollar_net_object: Keras sequential net collar_net_object object - neural net object
    '''
    collar_net_weights_filename = 'C2Dx8x3x3xvalidx3x32x32XAxreluXC2Dx8x3x3xvalidXAxreluXMPx2x2XC2Dx16x3x3xvalidXAxreluXC2Dx16x3x3xvalidXAxreluXMPx2x2XFXDx256XAxreluXDx64XAxreluXDx5XAxsoftmaxXOPxsgdx4e-6x1e-6x0.9x1XCOxcc.hdf5'
    ccollar_net_object = build_sequential_net_from_weights_filename(collar_net_weights_filename)
    return ccollar_net_object

def collar_classifier(collar_images, collar_net_object):
    '''
    :param collar_images: openCV image array of BGR channels, size 32x32 of collar area
    :param collar_collar_net_object: the appropriate net object, with the collar classifying buildup
    :return: classes_vector - a vector which sum is 1.0, and convey the probability of each collar type, i.e. []

    '''

    if len(np.array(collar_images).shape) > 4:
        result = []
        for image in collar_images:
            size_batch = len(image)
            proba = collar_net_object.predict_proba(image, batch_size=size_batch)
            classes = collar_net_object.predict_classes(image, batch_size=size_batch)
            # score = collar_net_object.evaluate(X_train, Y_train, batch_size=size_batch)
            # print proba
            max_val_location = np.where(proba == np.amax(proba))
            max_values_in_each_category = proba[max_val_location[0], :][0]#[np.amax(proba[:, 0]), np.amax(proba[:, 1]), np.amax(proba[:, 2]), np.amax(proba[:, 3]), np.amax(proba[:, 4])]
            print max_values_in_each_category
            # max_values_in_each_category = [np.mean(proba[:, 0]), np.mean(proba[:, 1]), np.mean(proba[:, 2]), np.mean(proba[:, 3]), np.mean(proba[:, 5])]
            # category_index = np.argmax(max_values_in_each_category, axis=0)
            res = {'crewneck' : max_values_in_each_category[0], 'roundneck' : max_values_in_each_category[1], 'scoopneck' : max_values_in_each_category[2], 'squareneck' : max_values_in_each_category[3], 'v-neck' : max_values_in_each_category[4]}
            result.append(res)
            print res
    else:
        size_batch = len(collar_images)
        proba = collar_net_object.predict_proba(collar_images, batch_size=size_batch)
        classes = collar_net_object.predict_classes(collar_images, batch_size=size_batch)
        # score = collar_net_object.evaluate(X_train, Y_train, batch_size=size_batch)
        # print proba
        max_val_location = np.where(proba == proba.max())
        print proba
        print max_val_location
        max_values_in_each_category = proba[max_val_location[0], :][0]#[np.amax(proba[:, 0]), np.amax(proba[:, 1]), np.amax(proba[:, 2]), np.amax(proba[:, 3]), np.amax(proba[:, 4])]
        # max_values_in_each_category = [np.mean(proba[:, 0]), np.mean(proba[:, 1]), np.mean(proba[:, 2]), np.mean(proba[:, 3]), np.mean(proba[:, 4])]
        # category_index = np.argmax(max_values_in_each_category, axis=0)
        result = {'crewneck' : max_values_in_each_category[0], 'roundneck' : max_values_in_each_category[1], 'scoopneck' : max_values_in_each_category[2], 'squareneck' : max_values_in_each_category[3], 'v-neck' : max_values_in_each_category[4]}
        print result

    # print result
    # print max_values_in_each_category
    # classes = classes[classes > 0]
    # print classes
    # category_index = int(mode(classes)[0])
    # print category_index
    #
    # print 'result of collar classifier for image is: ' + result[category_index]

    classes_vector = result

    return classes_vector


def collar_lab():
    collar_net = collar_classifier_net()
    image_to_test = [cv2.imread('/home/nate/Desktop/wild_v_3.jpg')]

    results = collar_classifier(image_to_test, collar_net)

    print results

collar_lab()
# build_sequential_net_from_weights_filename('C2Dx32x3x3xvalidx3x32x32XAxreluXMPx2x2XC2Dx16x3x3xvalidXAxreluXDOx0.25XC2Dx8x3x3xvalidXAxreluXDOx0.25XFXDx1152XAxreluXDOx05XDx5XAxsoftmaxXOPxsgdx1e-6x1e-6x0.9x1XCOxcc.hdf5')