import os
import pickle
import numpy as np
from scipy.stats import mode
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

def collar_images_maker_for_testing(image, face_box):

    collar_images = []
    a = 1.25 # scalar for increasing collar box in relation to face box (1==100%)
    max_angle = 15 # tilt angle of the image for diversification
    angle_offset = 5 # tilt angle of the image for diversification
    output_images_size = (32, 32) # pixels^2

    a = (a-1)/2
    # checking if the face (ancore) is present / detected:
    if len(face_box) == 0:
        return
    face = face_box
    face = face
    row, col, dep = image.shape
    if row < (face[1]+2*face[3])*(1+a):
        return collar_images
    # no flip along vertical axis:
    collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
    flipped_collar_image_center_point = (col - face[0]+0.5*face[2], face[1]+1.5*face[3])
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        image_of_collar = image_of_rotated_collar[(face[1]+face[3])*(1-a):
                            (face[1]+2*face[3])*(1+a),
                            (face[0])*(1-a):(face[0]+face[2])*(1+a)]
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        collar_images.append(resized_image_of_collar)

    # flip along vertical axis:
    image = np.fliplr(image)
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        image_of_collar = image_of_rotated_collar[(face[1]+face[3])*(1-a):
                            (face[1]+2*face[3])*(1+a),
                            col-((face[0]+face[2])*(1+a)):col-((face[0])*(1-a))]
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        collar_images.append(resized_image_of_collar)


    collar_images = np.array(collar_images)
    images_vector_shape = collar_images.shape
    collar_images = np.reshape(collar_images, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))
    return collar_images


def collar_classifier_neural_net(collar_images):

    if len(collar_images) == 0:
        print 'no face, so no collar detected.'
        return
    size_batch = len(collar_images)

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
    model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(Activation('hard_sigmoid'))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('hard_sigmoid'))
    model.add(Dense(64))
    model.add(Activation('hard_sigmoid'))
    model.add(Dense(3))
    model.add(Activation('softmax'))


    optimizer_method = Adadelta()#SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method)
    # model.load_weights(pickle.load(open('model_weights_pickled')))
    model.load_weights('collar_CNN.pymodel_weights_whatever.hdf5')
    proba = model.predict_proba(collar_images, batch_size=size_batch)
    classes = model.predict_classes(collar_images, batch_size=32)

    max_values_in_each_category = [np.amax(proba[:, 0]), np.amax(proba[:, 1]), np.amax(proba[:, 2])]
    # max_values_in_each_category = [np.mean(proba[:, 0]), np.mean(proba[:, 1]), np.mean(proba[:, 2])]
    # category_index = np.argmax(max_values_in_each_category, axis=0)
    result = {'roundneck' : max_values_in_each_category[0], 'squareneck' : max_values_in_each_category[1], 'v-neck' : max_values_in_each_category[2]}
    # print result

    # print max_values_in_each_category
    # classes = classes[classes > 0]
    # print classes
    # category_index = int(mode(classes)[0])
    # print category_index
    #
    # print 'result of collar classifier for image is: ' + result[category_index]
    return result


def collar_classifier(image, face_box):
    collar_images = collar_images_maker_for_testing(image, face_box)
    return collar_classifier_neural_net(collar_images)

