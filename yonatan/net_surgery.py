import caffe
import os
import argparse
import cv2
import numpy as np
import logging

net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/resnet_152_dress_texture/ResNet-152-deploy.prototxt', '/data/yonatan/yonatan_caffemodels/dressTexture_caffemodels/caffe_resnet152_snapshot_dress_texture_10_categories_iter_2500.caffemodel', caffe.TEST)

params = {}

all_params = [p for p in net.params ]
for layer in all_params: #loops over different layers
    for i in range(len(net.params[layer])): #loops over stuff in layer
        #for instance there may be weights and biases , or just weights
        print "len(net.params[layer]): {0}".format(len(net.params[layer]))
        params[i] = net.params[layer][i].data




        # print "net.params[layer][i].data: {0}".format(net.params[layer][i].data[0])
        # print "net.params[layer][i].data: {0}".format(net.params[layer][i].data[95])
        # print "I'm here!!!"
        # print "net.params[layer][i].data: {0}".format(net.params[layer][i].data[0])