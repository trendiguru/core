
import os
import numpy as np
import cv2
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def get_data(my_path):#my_path=os.path.dirname(os.path.abspath(__file__))):
    '''
    '''
    image_file_types = ['.jpg','.png','.bmp','.gif']
    only_files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path, f))]
    output_tag = []
    output_image = []

    # only_files = only_files[0:150]
    print len(only_files)
    for file_name in only_files:
        for image_type in image_file_types:
            if image_type in file_name:
                image = cv2.imread(my_path + file_name, 1)#np.uint8(
                output_image.append(image)
                output_tag.append(int(file_name[-5]))


    output_vector_size = max(output_tag)
    output_vector = []
    for tag in output_tag:
        zero_output_vector = np.zeros(output_vector_size, dtype=np.uint8)
        zero_output_vector[tag-1] = 1
        output_vector.append(zero_output_vector)

    output_image = np.array(output_image)
    output_vector = np.array(output_vector)

    images_vector_shape = output_image.shape
    output_image = np.reshape(output_image, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))
    # output_vector = np.reshape(output_vector)

    return output_image, output_vector


path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
output_image, output_vector = get_data(path)

print output_image.shape
print output_vector.shape

X_train = output_image
Y_train = output_vector







batch_size = 32
epoches_number = 100
overwrite_weights = True

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='full', input_shape=(3, 32, 32)))
model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Convolution2D(64, 3, 3, border_mode='valid'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

EarlyStopping(monitor='val_loss', patience=0, verbose=0)
# checkpointer = ModelCheckpoint(os.path.abspath(__file__) + 'weights.hdf5', verbose=1, save_best_only=True)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoches_number)#, callbacks=[checkpointer])
model.save_weights('model_weight.hdf5', overwrite_weights)

