import os
import pickle
import numpy as np
from scipy.stats import mode
import cv2
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

face_cascade = cv2.CascadeClassifier('/home/nate/Desktop/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')

def collar_images_maker_for_testing(image):

    collar_images = []
    image_file_types = ['.jpg', 'jpeg', '.png', '.bmp', '.gif']
    a = 1.35 # scalar for increasing collar box in relation to face box (1==100%)
    max_angle = 10 # tilt angle of the image for diversification
    angle_offset = 5 # tilt angle of the image for diversification
    output_images_size = (32, 32) # pixels^2

    # offset_range = np.arange(-max_offset, max_offset * 1.01, delta_offset)
    a = (a-1)/2
    if image == None:
        return collar_images
    face = face_cascade.detectMultiScale(image, 1.1, 2)#[0]
    # checking if the face (ancore) is present / detected:
    if len(face) == 0:
        return collar_images
    face = face[0]
    row, col, dep = image.shape
    if row < (face[1]+2*face[3])*(1+a):
        return collar_images
    offsetted_face = face
    # no flip along vertical axis:
    collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
    flipped_collar_image_center_point = (col - face[0]+0.5*face[2], face[1]+1.5*face[3])
    # for offset1 in offset_range:
    #     offsetted_face[0] = face[0] + offset1 * face[2]
    #     for offset2 in offset_range:
    #         offsetted_face[1] = face[1] + offset2 * face[3]
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        if len(image_of_rotated_collar) > 0:
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
            collar_images.append(resized_image_of_collar)
        # cv2.imshow('S', resized_image_of_collar)
        # cv2.waitKey(0)

    # flip along vertical axis:
    image = np.fliplr(image)
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        if len(image_of_rotated_collar) > 0:
            image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
            resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
            collar_images.append(resized_image_of_collar)
        # cv2.imshow('S', resized_image_of_collar)
        # cv2.waitKey(0)
#
    # image_of_collar = image[(face[1]+face[3])*(1-a):(face[1]+2*face[3])*(1+a),
    #                     (face[0])*(1-a):(face[0]+face[2])*(1+a)]
    # resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
    # collar_images.append(resized_image_of_collar)
#
    collar_images = np.array(collar_images)
    images_vector_shape = collar_images.shape
    collar_images = np.reshape(collar_images, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))

    return collar_images

def short_collar_images_maker_for_testing(image):
    collar_images = []
    image_file_types = ['.jpg', '.jpeg', '.png','.bmp','.gif']
    a = 1.35 # scalar for increasing collar box in relation to face box (1==100%)
    # max_angle = 10 # tilt angle of the image for diversification
    # angle_offset = max_angle/2 # tilt angle of the image for diversification
    # max_offset = 0.01 # maximum horizontal movement (% (out of box X) of the collar box for diversification
    # delta_offset = max_offset # horizontal movement increments (%)
    output_images_size = (32, 32) # pixels^2

    # offset_range = np.arange(-max_offset, max_offset * 1.01, delta_offset)
    a = (a-1)/2
    face = face_cascade.detectMultiScale(image, 1.1, 2)#[0]
    # checking if the face (ancore) is present / detected:
    if len(face) == 0:
        return
    face = face[0]
    offsetted_face = face
    row, col, dep = image.shape

    # no flip along vertical axis:
    collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
    flipped_collar_image_center_point = (col - face[0]+0.5*face[2], face[1]+1.5*face[3])
    # for offset1 in offset_range:
    #     offsetted_face[0] = face[0] + offset1 * face[2]
    #     for offset2 in offset_range:
    #         offsetted_face[1] = face[1] + offset2 * face[3]
    # for angle in range(-max_angle, max_angle+1, angle_offset):
    #     rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
    #     image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
    image_of_rotated_collar = image
    image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                        (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                        (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
    if len(image_of_collar) > 0:
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        collar_images.append(resized_image_of_collar)

    # flip along vertical axis:
    image = np.fliplr(image)
    # for angle in range(-max_angle, max_angle+1, angle_offset):
    #     rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
    #     image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
    image_of_rotated_collar = image
    image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                        (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                        col-((offsetted_face[0]+offsetted_face[2])*(1+a)):col-((offsetted_face[0])*(1-a))]
    if len(image_of_collar) > 0:
        resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
        collar_images.append(resized_image_of_collar)
#
    # image_of_collar = image[(face[1]+face[3])*(1-a):(face[1]+2*face[3])*(1+a),
    #                     (face[0])*(1-a):(face[0]+face[2])*(1+a)]
    # resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
    # collar_images.append(resized_image_of_collar)
#
    collar_images = np.array(collar_images)
    images_vector_shape = collar_images.shape
    collar_images = np.reshape(collar_images, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))
    return collar_images

def collar_classifier_neural_net(collar_images):

    if len(collar_images) == 0:
        print 'no face, so no collar detected.'
        return


    # model = Sequential()
    # model.add(Convolution2D(16, 3, 3, border_mode='full', input_shape=(3, 32, 32)))
    # model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(32, 3, 3))
    # model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(Activation('hard_sigmoid'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Convolution2D(256, 3, 3))
    # # model.add(Activation('hard_sigmoid'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.25))
    # # model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    # # model.add(Activation('hard_sigmoid'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    # # model.add(Activation('hard_sigmoid'))
    # # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('hard_sigmoid'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(128))
    # model.add(Activation('hard_sigmoid'))
    # model.add(Dense(64))
    # model.add(Activation('hard_sigmoid'))
    #
    # model.add(Dense(5))
    # model.add(Activation('softmax'))

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
    model.add(Activation('hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    # model.add(Activation('hard_sigmoid'))
    # model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    # model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('hard_sigmoid'))
    # model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Activation('hard_sigmoid'))
    # model.add(Dropout(0.25))
    model.add(Dense(5))
    model.add(Activation('softmax'))




    optimizer_method = Adadelta()#SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method)
    # model.load_weights(pickle.load(open('model_weights_pickled')))
    model.load_weights('3x32c_1024_128_5_ff.hdf5')

    if len(np.array(collar_images).shape) > 4:
        result = []
        for image in collar_images:
            size_batch = len(image)
            proba = model.predict_proba(image, batch_size=size_batch)
            classes = model.predict_classes(image, batch_size=size_batch)
            # score = model.evaluate(X_train, Y_train, batch_size=size_batch)
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
        proba = model.predict_proba(collar_images, batch_size=size_batch)
        classes = model.predict_classes(collar_images, batch_size=size_batch)
        # score = model.evaluate(X_train, Y_train, batch_size=size_batch)
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

    return result


# image = cv2.imread('/home/nate/Desktop/wild_v_2.jpg', 1)
#
# collar_images = collar_images_maker_for_testing(image)
# collar_classifier_neural_net(collar_images)
# collar_images = short_collar_images_maker_for_testing(image)
# collar_classifier_neural_net(collar_images)
