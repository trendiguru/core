import caffe
import os
import argparse
import cv2
import numpy as np
from numpy import linalg as LA
import logging

net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/alexnet/Alexnet_deploy.prototxt', '/data/yonatan/yonatan_caffemodels/genderator_caffemodels/caffe_alexnet_train_faces_iter_100000.caffemodel', caffe.TEST)
# net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/resnet18/deploy.prototxt', '/data/yonatan/yonatan_files/prepared_caffemodels/ResNet-18-model.caffemodel', caffe.TEST)
# net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/resnet_50_gender_by_face/ResNet-50-deploy.prototxt', '/data/yonatan/yonatan_caffemodels/genderator_caffemodels/caffe_resnet50_snapshot_sgd_genfder_by_face_iter_10000.caffemodel', caffe.TEST)

params = {}
layer_data = {}
new_caffemodel = {}

all_params = [p for p in net.params]
print all_params
for num_layer, layer in enumerate(all_params):  # loops over different layers
    params = {}
    new_params = {}
    print "layer: {0}".format(layer)
    print "len(net.params[layer]): {0}".format(len(net.params[layer]))
    for i in range(len(net.params[layer])):  # loops over stuff in layer - there's only 2 things with same len() in each layer, the first are all the weights, the second maybe mean of every filter (..?)
        # for instance there may be weights and biases , or just weights
        if layer == 'conv1':
            print net.blobs['conv1'].data[0].shape

        params[i] = net.params[layer][i].data
        print "len(params[{0}]): {1}".format(i, len(params[i]))
        print "shape of layer: {0}".format(params[i].shape)

    print "net.blobs['{0}'].data[0].shape: {1}".format(layer, net.blobs[layer].data[0].shape)

    new_params = params.copy()

    for filter_num in range(len(params[0])):
        # print "filter_num: {0}".format(filter_num)
        filter_RGB = params[0][filter_num]
        if type(filter_RGB) != np.float32:
            # print "len(filter_RGB): {0}".format(len(filter_RGB))
            # print "filter_RGB: {0}".format(filter_RGB)
            # print "filter: {0}".format(filter_RGB)
            filter_R = filter_RGB[0]
            filter_G = filter_RGB[1]
            filter_B = filter_RGB[2]
            norm_R = LA.norm(filter_RGB[0])
            # print "filter_R: {0}".format(filter_R)
            # print "norm_R: {0}".format(norm_R)
            norm_G = LA.norm(filter_RGB[1])
            norm_B = LA.norm(filter_RGB[2])
            new_params[0][filter_num][0] = filter_R / norm_R  # new_filter_R
            new_params[0][filter_num][1] = filter_G / norm_G  # new_filter_G
            new_params[0][filter_num][2] = filter_B / norm_B  # new_filter_B
    print "type(filter_RGB): {0}".format(type(filter_RGB))

    print "new layer!"
    layer_data[num_layer] = params
    new_caffemodel[num_layer] = new_params
    # break

## layer_data[0] - gives me the first layer full data (net.params[layer][0].data and net.params[layer][1].data)
## layer_data[0][0] - gives me all filters in first layer (only net.params[layer][0].data)
## layer_data[0][0][0] - gives me the first filter in the first layer (there's 3 filters in each filter - RGB
## layer_data[0][0][0][0] - gives me the Red filter of the first filter of the first layer


