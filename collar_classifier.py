import os
import numpy as np
import cv2
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

current_directory_name = os.getcwd()
weights_file_path = current_directory_name + '/saved_weight.hdf5'


def collar_set_creator(image):

    collar_set = []
    face_cascade = cv2.CascadeClassifier('/home/nate/Desktop/TrendiGuru/core/classifier_stuff/classifiers_to_test/face/haarcascade_frontalface_default.xml')
    a = 1.25 # scalar for increasing collar box in relation to face box (1==100%)
    max_angle = 5 # tilt angle of the image for diversification
    angle_offset = 5 # tilt angle of the image for diversification
    max_offset = 0.01 # maximum horizontal movement (% (out of box X) of the collar box for diversification
    delta_offset = max_offset # horizontal movement increments (%)
    output_images_size = (32, 32) # pixels^2

    offset_range = np.arange(-max_offset, max_offset * 1.01, delta_offset)
    a = (a-1)/2
    face = face_cascade.detectMultiScale(image, 1.3, 5)#[0]

    # checking if the face (ancore) is present / detected:
    if len(face) == 0:
        return collar_set
    face = face[0]
    offsetted_face = face
    row, col, dep = image.shape

    # no flip along vertical axis:
    collar_image_center_point = (face[0]+0.5*face[2], face[1]+1.5*face[3])
    image = np.fliplr(image)
    flipped_collar_image_center_point = (col - face[0]+0.5*face[2], face[1]+1.5*face[3])
    for offset1 in offset_range:
        offsetted_face[0] = face[0] + offset1 * face[2]
        for offset2 in offset_range:
            offsetted_face[1] = face[1] + offset2 * face[3]
            for angle in range(-max_angle, max_angle+1, angle_offset):
                rotated_image_matrix = cv2.getRotationMatrix2D(collar_image_center_point, angle, 1.0)
                image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
                image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                    (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                    (offsetted_face[0])*(1-a):(offsetted_face[0]+offsetted_face[2])*(1+a)]
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                collar_set.append(resized_image_of_collar)

            # flip along vertical axis:
            for angle in range(-max_angle, max_angle+1, angle_offset):
                rotated_image_matrix = cv2.getRotationMatrix2D(flipped_collar_image_center_point, angle, 1.0)
                image_of_rotated_collar = cv2.warpAffine(image, rotated_image_matrix,(row, col))
                image_of_collar = image_of_rotated_collar[(offsetted_face[1]+offsetted_face[3])*(1-a):
                                    (offsetted_face[1]+2*offsetted_face[3])*(1+a),
                                    col-((offsetted_face[0]+offsetted_face[2])*(1+a)):col-((offsetted_face[0])*(1-a))]
                resized_image_of_collar = cv2.resize(image_of_collar, output_images_size)
                collar_set.append(resized_image_of_collar)

    collar_set = np.array(collar_set)
    images_vector_shape = collar_set.shape
    collar_set = np.reshape(collar_set, (images_vector_shape[0], images_vector_shape[3],
                                         images_vector_shape[1], images_vector_shape[2]))

    return collar_set


def images_batch_collar_classification(images_batch, weights_file_path):

    length_of_images_batch = len(images_batch)

    model = Sequential()
    model.add(Convolution2D(9, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
    model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(9, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(9, 3, 3, border_mode='valid'))
    model.add(Activation('hard_sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(9, 3, 3))
    model.add(Activation('hard_sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('hard_sigmoid'))
    # model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    optimizer_method = Adadelta()#SGD(lr=0.00000001, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_method)
    model.load_weights(weights_file_path)


    # for image in images_batch:
    classes = model.predict_classes(images_batch, batch_size=length_of_images_batch)
    proba = model.predict_proba(images_batch, batch_size=length_of_images_batch)

    print classes
    print proba





collar_set = collar_set_creator(cv2.imread('/home/nate/Desktop/TrendiGuru/core/collar_neural_net/crewneck_17.png', 1))
print len(collar_set)
images_batch_collar_classification(collar_set, weights_file_path)