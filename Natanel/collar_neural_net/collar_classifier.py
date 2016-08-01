import os
import pickle
import numpy as np
from scipy.stats import mode
import cv2
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.layers.normalization import BatchNormalization
import time


def collar_images_maker_for_testing(image, face_box):

    collar_images = []
    a = 1.5 # scalar for increasing collar box in relation to face box (1==100%)
    max_angle = 15 # tilt angle of the image for diversification
    angle_offset = 5 # tilt angle of the image for diversification
    output_images_size = (48, 48) # pixels^2

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
            image_of_collar = image_of_rotated_collar[(face[1]+face[3])*(1-a):(face[1]+2.1*face[3])*(1+a),
                                (face[0])*(1-a):(face[0]+face[2])*(1+a)]
            resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
            collar_images.append(resized_image_of_collar)

    # flip along vertical axis:
    image = np.fliplr(image)
    for angle in range(-max_angle, max_angle+1, angle_offset):
        rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
        image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
        if len(image_of_rotated_collar) > 0:
            image_of_collar = image_of_rotated_collar[(face[1]+face[3])*(1-a):(face[1]+2.1*face[3])*(1+a),
                                (face[0])*(1-a):(face[0]+face[2])*(1+a)]
            resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
            collar_images.append(resized_image_of_collar)


    collar_images = np.array(collar_images)
    images_vector_shape = collar_images.shape
    collar_images = np.reshape(collar_images, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))
    return collar_images


def net(input_shape):
    input_img = Input(input_shape)
    # convolutional layers: (N_kernels, h_kernel, W_kernel)
    conv = BatchNormalization()(input_img)
    conv = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    # conv = Dropout(0.25)(conv)
    conv = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Dropout(0.1)(conv)
    conv = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Dropout(0.15)(conv)
    conv = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv)
    conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Dropout(0.2)(conv)
    conv = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(conv)
    # conv = MaxPooling2D((2, 2), strides=(2, 2))(conv)
    conv = Dropout(0.25)(conv)
    fc = Flatten()(conv)

    # fully connected layers: (N_newrons)
    # fc = Dense(16, activation='relu')(fc)
    # # fc = Dropout(0.25)(fc)
    fc = Dense(5, activation='softmax')(fc)
    model = Model(input=input_img, output=fc)
    optimizer_method = 'adam'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method, metrics=['accuracy'])
    model.load_weights(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collar_model_weights.hdf5'))

    return model


def collar_classifier_neural_net(model, collar_images):

    if len(collar_images) == 0:
        print 'no face, so no collar detected.'
        return
    size_batch = len(collar_images)

    collar_images = np.array(collar_images, dtype='float32') / 255
    collar_images = np.transpose(collar_images, (0, 3, 1, 2))
    proba = model.predict_proba(collar_images, batch_size=size_batch)
    # classes = model.predict_classes(collar_images, batch_size=32)

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


def collar_classifier(model, image, face_box):
    collar_images = collar_images_maker_for_testing(image, face_box)
    return collar_classifier_neural_net(model, collar_images)


def collar_distance(collar1_vec, collar2_vec, weights_vec):
    # alll inputs must be of similar shape numpy float32 arrays
    dC12 = cv2.compareHist(collar1_vec * weights_vec,
                           collar2_vec * weights_vec, cv2.HISTCMP_CHISQR)
    return dC12




# if __name__ == "__main__":
#     img_arr = cv2.imread('images/vneck.jpg')
# #    img_arr = cv2.imread('images/roundneck.jpg')
# #    img_arr = cv2.imread('images/squareneck.jpg')
#     face_bbs = background_removal.find_face_cascade(img_arr, 10)
#     if face_bbs['are_faces'] is True:
#         print('face bbs:'+str(face_bbs))
#         first_bb = face_bbs['faces'][0]
#         print(first_bb)
#         t0 = time.time()
#         res = collar_classifier(img_arr, first_bb)
#         print('res = '+str(res)+' time elapsed:'+str(time.time()-t0))
#     else:
#         print('no face found')



