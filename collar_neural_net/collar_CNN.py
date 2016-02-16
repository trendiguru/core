
import os
import pickle
import numpy as np
import cv2
import h5py
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

def get_data(my_path):#, testing_amount=0.2):#my_path=os.path.dirname(os.path.abspath(__file__))):
    '''
    '''
    # if testing_amount > 1.0 or testing_amount < 0.0:
    #     print 'testing_amount should be between 0 and 1 (float)!'
    #     return
    image_file_types = ['.jpg', '.jpeg', '.png','.bmp','.gif']
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
    print output_vector_size
    output_vector = []
    # amount_of_each_tag = np.zeros(output_vector_size)
    for tag in output_tag:
        zero_output_vector = np.zeros(output_vector_size, dtype=np.uint8)
        if tag > 0:
           zero_output_vector[tag-1] = 1
           # amount_of_each_tag[tag-1] += 1
        output_vector.append(zero_output_vector)

    output_image = np.array(output_image)
    output_vector = np.array(output_vector)

    images_vector_shape = output_image.shape
    output_image = np.reshape(output_image, (images_vector_shape[0], images_vector_shape[3],
                                             images_vector_shape[1], images_vector_shape[2]))
    # print amount_of_each_ta g
    # testing_input = []
    # testing_output = []
    # training_inputnano = []
    # training_output = []
    # for type in range(output_vector_size):
    #     amount_of_tag = int(amount_of_each_tag[type] * testing_amount)
    #     testing_input = testing_input.append(output_image[0:amount_of_tag, :, :, :])
    #     testing_output = testing_output.append(output_vector[0:amount_of_tag, :])
    #     training_input = training_input.append(output_image[amount_of_tag:, :, :, :])
    #     training_output = training_output.append(output_vector[amount_of_tag:, :])
    #
    # return testing_input, testing_output, training_input, training_output
    return output_image, output_vector

path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
# testing_input, testing_output, training_input, training_output = get_data(path)
#
# print testing_input.shape, testing_output.shape
# print training_input.shape, training_output.shape
#
# X_train = training_input
# Y_train = training_output
# X_test = testing_input
# Y_test = testing_output

output_image, output_vector = get_data(path)
X_train = output_image
Y_train = output_vector



model_description = 'EX' #'32k5x5CV1_2x2MP1_32k3x3CV2_32k3x3CV3_32k3x3CV4_2x2MP2_64dFC1_3dFC2'
size_batch = 32
epoches_number = 10000
overwrite_weights = True
testing_amount = 0.05

model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))


optimizer_method = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)#Adagrad()#Adadelta()#RMSprop()#Adam()#Adadelta()#
model.compile(loss='categorical_crossentropy', optimizer=optimizer_method)

EarlyStopping(monitor='val_loss', patience=0, verbose=1)
checkpointer = ModelCheckpoint('best_model_weights_' + model_description + '.hdf5', verbose=1, save_best_only=True)


model.fit(X_train, Y_train, batch_size=size_batch, nb_epoch=epoches_number, validation_split=testing_amount, show_accuracy=True, callbacks=[checkpointer])
score = model.evaluate(X_train, Y_train, batch_size=size_batch)


json_string = model.to_json()
open('model_architecture_' + model_description + '.json', 'w').write(json_string)
model.save_weights('model_weights_' + model_description + '.hdf5', overwrite_weights)

layer_weights = [layer.get_weights() for layer in model.layers]
with open('model_weights_pickled', 'w') as weights_file:
    pickle.dump(layer_weights, weights_file)

