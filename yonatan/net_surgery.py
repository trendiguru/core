import caffe
import os
import argparse
import cv2
import numpy as np
import logging

net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/alexnet/Alexnet_deploy.prototxt', '/data/yonatan/yonatan_caffemodels/genderator_caffemodels/caffe_alexnet_train_faces_iter_100000.caffemodel', caffe.TEST)

params = {}

all_params = [p for p in net.params ]
for layer in all_params: #loops over different layers
    for i in range(len(net.params[layer])): #loops over stuff in layer - there's only 2 things with same len() in each layer, the first are all the weights, the second maybe mean of every filter (..?)
        #for instance there may be weights and biases , or just weights
        print "layer: {0}".format(layer)
        print "len(net.params[layer]): {0}".format(len(net.params[layer]))
        params[i] = net.params[layer][i].data
    print "new layer!"
    # break





        # print "net.params[layer][i].data: {0}".format(net.params[layer][i].data[0])
        # print "net.params[layer][i].data: {0}".format(net.params[layer][i].data[95])
        # print "I'm here!!!"
        # print "net.params[layer][i].data: {0}".format(net.params[layer][i].data[0])