import caffe
import os
import argparse
import cv2
import numpy as np
from numpy import linalg as LA
import logging

net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/alexnet/Alexnet_deploy.prototxt', '/data/yonatan/yonatan_caffemodels/genderator_caffemodels/caffe_alexnet_train_faces_iter_100000.caffemodel', caffe.TEST)

params = {}
layer_data = {}

all_params = [p for p in net.params]
for num_layer, layer in enumerate(all_params): #loops over different layers
    params = {}
    print "layer: {0}".format(layer)
    for i in range(len(net.params[layer])): #loops over stuff in layer - there's only 2 things with same len() in each layer, the first are all the weights, the second maybe mean of every filter (..?)
        #for instance there may be weights and biases , or just weights
        # if layer == 'conv3':
        print "len(net.params[layer]): {0}".format(len(net.params[layer]))
        params[i] = net.params[layer][i].data
    print "new layer!"
    layer_data[num_layer] = params
    # break


## layer_data[0] - gives me the first layer full data (net.params[layer][0].data and net.params[layer][1].data)
## layer_data[0][0] - gives me all filters in first layer (only net.params[layer][0].data)
## layer_data[0][0][0] - gives me the first filter in the first layer (there's 3 filters in each filter - RGB
## layer_data[0][0][0][0] - gives me the Red filter of the first filter of the first layer
