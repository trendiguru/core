import caffe
import os
import argparse
import cv2
import numpy as np
import logging

net = caffe.Net('/data/yonatan/yonatan_files/trendi/yonatan/alexnet/Alexnet_deploy.prototxt', '/data/yonatan/yonatan_caffemodels/genderator_caffemodels/caffe_alexnet_train_faces_iter_100000.caffemodel', caffe.TEST)

caffe.set_device(1)
caffe.set_mode_gpu()

params = 0

all_params = [p for p in net.params ]
for layer in all_params: #loops over different layers
    for i in range(len(net.params[layer])): #loops over stuff in layer
        #for instance there may be weights and biases , or just weights
        params[i] = net.params[layer][i].data
        print params[i]
    break
