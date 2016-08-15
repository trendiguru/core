# coding: utf-8
__author__ = 'jeremy'
import socket
from pylab import *
from trendi.classifier_stuff.caffe_nns import lmdb_utils
import sys
import caffe
import caffe.draw
from trendi.utils import imutils
from matplotlib import pyplot as plt
import numpy as np
import os
from trendi import Utils
from caffe.proto import caffe_pb2
from google.protobuf import text_format
#sys.path.insert(0, 'python/')


try:
    import caffe
    from caffe import layers as L
    from caffe import params as P
except:
    print(sys.path)
    sys.path.append('/home/jr/sw/caffe/python')
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+'/home/jr/sw/caffe/python'
    import caffe
    from caffe import layers as L
    from caffe import params as P

#label = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_label', transform_param=dict(scale=1./255), ntop=1)
#data = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_data', transform_param=dict(scale=1./255), ntop=1)


def write_prototxt(proto_filename,test_iter = 9,solver_mode='GPU'):
    # The train/test net protocol buffer definition

    dir = os.path.dirname(proto_filename)
    filename = os.path.basename(proto_filename)
    file_base = filename.split('prototxt')[0]
    train_file = os.path.join(dir,file_base+'train.prototxt')
    test_file = os.path.join(dir,file_base + 'test.prototxt')
    # test_iter specifies how many forward passes the test should carry out. test_iter*batch_size<= # test images
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    # test_interval - Carry out testing every 500 training iterations.
    # base_lr - The base learning rate, momentum and the weight decay of the network.
    # lr_policy - The learning rate policy
    # display - Display every n iterations
    # max_iter - The maximum number of iterations
    # snapshot - snapshot intermediate results
    # snapshot prefix - dir for snapshot  - maybe requires '/' at end?
    # solver_mode - CPU or GPU
    prototxt ={ 'train_net':train_file,
                        'test_net': test_file,
                        'test_iter': test_iter,
                        'test_interval': 500,
                        'base_lr': 0.01,
                        'momentum': 0.9,
                        'weight_decay': 0.0005,
                        'lr_policy': "step",
                        'gamma': 0.1,
#                        'power': 0.75,
                        'display': 50,
                        'max_iter': 150000,
                        'snapshot': 5000,
                        'snapshot_prefix': 'snapshot/trainsharp_',
                        'solver_mode':solver_mode }

    print('writing prototxt:'+str(prototxt))
    with open(proto_filename,'w') as f:
        for key, val in prototxt.iteritems():
            line=key+':'+str(val)+'\n'
            if isinstance(val,basestring) and key is not 'solver_mode':
                line=key+':\"'+str(val)+'\"\n'
            f.write(line)


def examples(lmdb, batch_size):  #test_iter * batch_size <= n_samples!!!
    '''
    examples of python creation of prototxt components
    The net is returned as an object which can be stringified and written to a prototxt file along the lines of
        with open('examples/mnist/lenet_auto_train.prototxt','w') as f:
            f.write(str(lenet('examples/mnist/mnist_train_lmdb',64)))

    '''
    lr_mult1 = 1
    lr_mult2 = 2
    n=caffe.NetSpec()

    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,transform_param=dict(scale=1./255),ntop=2)
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=[meanB,meanG,meanR]),ntop=2)

    n.conv1 = L.Convolution(n.data,  kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
    n.conv2 = L.Convolution(n.pool1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride=1,num_output=20,weight_filler=dict(type='xavier'))
    n.conv1 = L.Convolution(n.data,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)])
    n.conv1_7x7_s2_3= L.Convolution(n.data_1,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=64,
                            pad = 3,
                            kernel_size=7,
                            stride = 2,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.conv =  L.Convolution(bottom, param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
        kernel_h=kh, kernel_w=kw, stride=stride, num_output=nout, pad=pad,
        weight_filler=dict(type='gaussian', std=0.1, sparse=sparse),
        bias_filler=dict(type='constant', value=0))

    # NOT TESTED.  padding is removed from the output rather than added to the input, and stride results in upsampling rather than downsampling
    # http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DeconvolutionLayer.html
    n.deconv = L.Deconvolution(n.bottom,
                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=64,
                            pad = 3,
                            kernel_size=7,
                            stride = 2,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))

    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.ip1 = L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.ip1 = L.InnerProduct(n.pool2,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=500,weight_filler=dict(type='xavier'))

    n.relu1 = L.ReLU(n.ip1, in_place=True)

    n.accuracy = L.Accuracy(n.ip2,n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2,n.label)

    n.pool1_norm1_6 = L.LRN(n.pool1_3x3_s2_5,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))

    n.conv2_norm2_11 = L.LRN(n.conv2_3x3_9,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))
    n.inception_3a_output_26 = L.Concat(bottom=[n.inception_3a_1x1_13,n.inception_3a_3x3_17,n.inception_3a_5x5_21,n.inception_3a_pool_proj_24])
    n.final_dropout = L.Dropout(n.inception_3a_avg_pool, dropout_param=dict(dropout_ratio=0.4),in_place=True)

    bottom_layers = [n.inception_3a_1x1_13,n.inception_3a_3x3_17,n.inception_3a_5x5_21,n.inception_3a_pool_proj_24]
    n.inception_3a_output_26 = L.Concat(*bottom_layers)

    n.loss = L.EuclideanLoss(n.output_layer,n.label)
    return n.to_proto()

def conv_relu(bottom,lr_mult1 = 1,lr_mult2 = 2,decay_mult1=1,decay_mult2 =0,n_output=64,pad=3,kernel_size=3,stride=1,weight_filler='xavier',bias_filler='constant',bias_const_val=0.2):
    conv = L.Convolution(bottom,
                        param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                        num_output=n_output,
                        pad = pad,
                        kernel_size=kernel_size,
                        stride = stride,
                        weight_filler=dict(type=weight_filler),
                        bias_filler=dict(type=bias_filler,value=bias_const_val))
    relu = L.ReLU(conv, in_place=True)
    return conv,relu

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def vgg16(db,mean_value=[112.0,112.0,112.0]):
    '''
    see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt
    :param db:
    :param mean_value:
    :return:
    '''
    #pad to keep image size if S=1 : p=(F-1)/2    , (W-F+2P)/S + 1  neurons in a layer   w:inputsize, F:kernelsize, P: padding, S:stride
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0
    batch_size = 1
    n=caffe.NetSpec()

    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=mean_value,mirror=True),ntop=2)

    n.conv1_1,n.relu1_1 = conv_relu(n.data,n_output=64,kernel_size=3,pad=1)
    n.conv1_2,n.relu1_2 = conv_relu(n.conv1_1,n_output=64,kernel_size=3,pad=1)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2_1,n.relu2_1 = conv_relu(n.pool1,n_output=128,kernel_size=3,pad=1)
    n.conv2_2,n.relu2_2 = conv_relu(n.conv2_1,n_output=128,kernel_size=3,pad=1)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv3_1,n.relu3_1 = conv_relu(n.pool2,n_output=256,kernel_size=3,pad=1)
    n.conv3_2,n.relu3_2 = conv_relu(n.conv3_1,n_output=256,kernel_size=3,pad=1)
    n.conv3_3,n.relu3_3 = conv_relu(n.conv3_2,n_output=256,kernel_size=3,pad=1)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv4_1,n.relu4_1 = conv_relu(n.pool3,n_output=512,kernel_size=3,pad=1)
    n.conv4_2,n.relu4_2 = conv_relu(n.conv4_1,n_output=512,kernel_size=3,pad=1)
    n.conv4_3,n.relu4_3 = conv_relu(n.conv4_2,n_output=512,kernel_size=3,pad=1)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv5_1,n.relu5_1 = conv_relu(n.pool4,n_output=512,kernel_size=3,pad=1)
    n.conv5_2,n.relu5_2 = conv_relu(n.conv5_1,n_output=512,kernel_size=3,pad=1)
    n.conv5_3,n.relu5_3 = conv_relu(n.conv5_2,n_output=512,kernel_size=3,pad=1)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.fc6 = L.InnerProduct(n.pool5,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.fc6, dropout_param=dict(dropout_ratio=0.5),in_place=True)

    n.fc7 = L.InnerProduct(n.fc6,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.5),in_place=True)

    n.fc8 = L.InnerProduct(n.fc7,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=1000)
    return n.to_proto()

def sharpmask(db,mean_value=[112.0,112.0,112.0]):
    '''
    see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt
    :param db:
    :param mean_value:
    :return:
    '''
    #pad to keep image size if S=1 : p=(F-1)/2    , (W-F+2P)/S + 1  neurons in a layer   w:inputsize, F:kernelsize, P: padding, S:stride
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0
    batch_size = 1
    n=caffe.NetSpec()
    #assuming input of size 224x224, ...
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=mean_value,mirror=True),ntop=2)

    n.conv1_1,n.relu1_1 = conv_relu(n.data,n_output=64,kernel_size=3,pad=1)
    n.conv1_2,n.relu1_2 = conv_relu(n.conv1_1,n_output=64,kernel_size=3,pad=1)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 112x112
    n.conv2_1,n.relu2_1 = conv_relu(n.pool1,n_output=128,kernel_size=3,pad=1)
    n.conv2_2,n.relu2_2 = conv_relu(n.conv2_1,n_output=128,kernel_size=3,pad=1)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 56x56
    n.conv3_1,n.relu3_1 = conv_relu(n.pool2,n_output=256,kernel_size=3,pad=1)
    n.conv3_2,n.relu3_2 = conv_relu(n.conv3_1,n_output=256,kernel_size=3,pad=1)
    n.conv3_3,n.relu3_3 = conv_relu(n.conv3_2,n_output=256,kernel_size=3,pad=1)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 28x28
    n.conv4_1,n.relu4_1 = conv_relu(n.pool3,n_output=512,kernel_size=3,pad=1)
    n.conv4_2,n.relu4_2 = conv_relu(n.conv4_1,n_output=512,kernel_size=3,pad=1)
    n.conv4_3,n.relu4_3 = conv_relu(n.conv4_2,n_output=512,kernel_size=3,pad=1)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 14x14
    n.conv5_1,n.relu5_1 = conv_relu(n.pool4,n_output=512,kernel_size=3,pad=1)
    n.conv5_2,n.relu5_2 = conv_relu(n.conv5_1,n_output=512,kernel_size=3,pad=1)
    n.conv5_3,n.relu5_3 = conv_relu(n.conv5_2,n_output=512,kernel_size=3,pad=1)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 7x7
    n.conv6_1,n.relu6_1 = conv_relu(n.pool5,n_output=4096,kernel_size=7,pad=3)
       #instead of L.InnerProduct(n.pool5,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.drop6 = L.Dropout(n.conv6_1, dropout_param=dict(dropout_ratio=0.5),in_place=True)

    n.conv6_2,n.relu6_2 = conv_relu(n.conv6_1,n_output=4096,kernel_size=7,pad=3)
        #instead of n.fc7 = L.InnerProduct(n.fc6,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.5),in_place=True)

    n.conv6_1,n.relu6_1 = conv_relu(n.pool5,n_output=4096,kernel_size=7,pad=3)

    n.deconv1 = L.Deconvolution(n.bottom,
                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                            num_output=64,
                            pad = 3,
                            kernel_size=7,
                            stride = 2,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))

    return n.to_proto()

'''layer {
  name: "data"
  type: "Python"
  top: "data"
  top: "label"
  python_param {
    module: "jrlayers"
    layer: "JrPixlevel"
    param_str: "{\'images_and_labels_file\': \'/home/jeremy/image_dbs/colorful_fashion_parsing_data/images_and_labelsfile_train.txt\', \'mean\': (104.0, 116.7, 122.7),\'augment\':True,\'augment_crop_size\':(224,224), \'batch_size\':9 }"
#    param_str: "{\'images_dir\': \'/home/jeremy/image_dbs/colorful_fashion_parsing_data/images/train_u21_256x256\', \'labels_dir\':\'/home/jeremy/image_dbs/colorful_fashion_parsing_data/labels_256x256/\', \'mean\': (104.00699, 116.66877, 122.67892)}"
#    param_str: "{\'sbdd_dir\': \'../../data/sbdd/dataset\', \'seed\': 1337, \'split\': \'train\', \'mean\': (104.00699, 116.66877, 122.67892)}"
  }
}

layer {
  name: "upscore8"
  type: "Deconvolution"
  bottom: "fuse_pool3"
  top: "upscore8"
  param {
    lr_mult: 1
  }
  convolution_param {
    num_output: 21
    bias_term: false
    kernel_size: 16
    stride: 8
  }
}

'''

def unet(db,mean_value=[112.0,112.0,112.0]):
    '''
    see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt
    :param db:
    :param mean_value:
    :return:
    '''
    #pad to keep image size if S=1 : p=(F-1)/2    , (W-F+2P)/S + 1  neurons in a layer   w:inputsize, F:kernelsize, P: padding, S:stride
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0
    batch_size = 1
    n=caffe.NetSpec()
    #assuming input of size 224x224, these are 224x244 (/1)
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=mean_value,mirror=True),ntop=2)
#    n.data,n.label=L.Data(type='Python',python_param=dict(module='jrlayers',layer='JrPixlevel'),ntop=2)

    n.conv1_1,n.relu1_1 = conv_relu(n.data,n_output=64,kernel_size=3,pad=1)
    n.conv1_2,n.relu1_2 = conv_relu(n.conv1_1,n_output=64,kernel_size=3,pad=1)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 112x112 (/2)
    n.conv2_1,n.relu2_1 = conv_relu(n.pool1,n_output=128,kernel_size=3,pad=1)
    n.conv2_2,n.relu2_2 = conv_relu(n.conv2_1,n_output=128,kernel_size=3,pad=1)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 56x56 (original /4)
    n.conv3_1,n.relu3_1 = conv_relu(n.pool2,n_output=256,kernel_size=3,pad=1)
    n.conv3_2,n.relu3_2 = conv_relu(n.conv3_1,n_output=256,kernel_size=3,pad=1)
    n.conv3_3,n.relu3_3 = conv_relu(n.conv3_2,n_output=256,kernel_size=3,pad=1)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 28x28 (original /8)
    n.conv4_1,n.relu4_1 = conv_relu(n.pool3,n_output=512,kernel_size=3,pad=1)
    n.conv4_2,n.relu4_2 = conv_relu(n.conv4_1,n_output=512,kernel_size=3,pad=1)
    n.conv4_3,n.relu4_3 = conv_relu(n.conv4_2,n_output=512,kernel_size=3,pad=1)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 14x14 (original /16)
    n.conv5_1,n.relu5_1 = conv_relu(n.pool4,n_output=512,kernel_size=3,pad=1)
    n.conv5_2,n.relu5_2 = conv_relu(n.conv5_1,n_output=512,kernel_size=3,pad=1)
    n.conv5_3,n.relu5_3 = conv_relu(n.conv5_2,n_output=512,kernel_size=3,pad=1)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 7x7 (original /32)
    n.conv6_1,n.relu6_1 = conv_relu(n.pool5,n_output=512,kernel_size=7,pad=3)
       #instead of L.InnerProduct(n.pool5,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.drop6_1 = L.Dropout(n.conv6_1, dropout_param=dict(dropout_ratio=0.5),in_place=True)
    n.conv6_2,n.relu6_2 = conv_relu(n.conv6_1,n_output=1024,kernel_size=7,pad=3)
        #instead of n.fc7 = L.InnerProduct(n.fc6,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.drop6_2 = L.Dropout(n.conv6_2, dropout_param=dict(dropout_ratio=0.5),in_place=True)
    n.conv6_3,n.relu6_3 = conv_relu(n.conv6_2,n_output=1024,kernel_size=7,pad=3)

#the following will be 14x14  (original /16)
#deconv doesnt work from python , so these need to be changed by hand #
    n.deconv7 = L.Convolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    num_output=1024,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv7_1,n.relu7_1 = conv_relu(n.deconv7,n_output=512,kernel_size=2,pad=1)  #watch out for padsize here, make sure outsize is 14x14
    n.cat7 = L.Concat(bottom=[n.conv5_3, n.conv7_1])
    n.conv7_2,n.relu7_2 = conv_relu(n.cat7,n_output=1024,kernel_size=3,pad=1)
    n.conv7_3,n.relu7_3 = conv_relu(n.conv7_2,n_output=1024,kernel_size=3,pad=1)

    #the following will be 28x28  (original /8)
    n.deconv8 = L.Convolution(n.conv7_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    num_output=1024,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv8_1,n.relu8_1 = conv_relu(n.deconv8,n_output=512,kernel_size=2,pad=1)
    n.cat8 = L.Concat(bottom=[n.conv4_3, n.conv8_1])
    n.conv8_2,n.relu8_2 = conv_relu(n.cat8,n_output=512,kernel_size=3,pad=1)  #this is halving N_filters
    n.conv8_3,n.relu8_3 = conv_relu(n.conv8_2,n_output=512,kernel_size=3,pad=1)


    #the following will be 56x56  (original /4)
    n.deconv9 = L.Convolution(n.conv8_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    num_output=512,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv9_1,n.relu9_1 = conv_relu(n.deconv8,n_output=256,kernel_size=2,pad=1)
    n.cat9 = L.Concat(bottom=[n.conv3_3, n.conv8_1])
    n.conv9_2,n.relu9_2 = conv_relu(n.cat9,n_output=256,kernel_size=3,pad=1)  #this is halving N_filters
    n.conv9_3,n.relu9_3 = conv_relu(n.conv9_2,n_output=256,kernel_size=3,pad=1)


    #the following will be 112x112  (original /2)
    n.deconv3 = L.Convolution(n.conv9_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=512,pad = 0,kernel_size=2,stride = 2,
                            weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))

    #the following will be 224x224  (original /1)
    n.deconv4 = L.Convolution(n.deconv3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=512,pad = 0,kernel_size=2,stride = 2,
                            weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))


    n.loss = L.SoftmaxWithLoss(n.deconv4, n.label)

#    n.deconv1 = L.Deconvolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                convolution_param=[dict(num_output=512,bias_term=False,kernel_size=2,stride=2)])
    return n.to_proto()



''' #

  convolution_param {
    num_output: 21
    bias_term: false
    kernel_size: 16
    stride: 8
  }


'''

def display_conv_layer(blob):
    print('blob:'+str(blob))
    print('blob data size:{}'.format(blob.data.shape))
    plt.imshow(blob.data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray',block=False)
    plt.show(block=False)
#    plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray',block=False)
#    plt.show(block=False)
#    print solver.net.blobs['label'].data[:8]

def draw_net(prototxt,outfile):
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt).read(), net)
    print('Drawing net to %s' % outfile)
    caffe.draw.draw_net_to_file(net, outfile, 'TB') #TB for vertical, 'RL' for horizontal

def run_net(net_builder,nn_dir,train_db,test_db,batch_size = 64,n_classes=11,meanB=None,meanG=None,meanR=None,n_filters=50,n_ip1=1000,n_test_items=None):
    if host == 'jr-ThinkPad-X1-Carbon':
        pc = True
        solver_mode = 'CPU'
        caffe.set_mode_cpu()
    else:
        pc = False
        solver_mode = 'GPU'
        caffe.set_mode_gpu()
        caffe.set_device(0)

    Utils.ensure_dir(nn_dir)
    proto_filename = 'my_solver.prototxt'
    proto_file_path = os.path.join(nn_dir,'my_solver.prototxt')
    test_iter = 100
    write_prototxt(proto_file_path,test_iter = test_iter,solver_mode=solver_mode)
    proto_file_base = proto_filename.split('prototxt')[0]
    train_protofile = os.path.join(nn_dir,proto_file_base+'train.prototxt')
    test_protofile = os.path.join(nn_dir,proto_file_base+'test.prototxt')
    deploy_protofile = os.path.join(nn_dir,proto_file_base+'deploy.prototxt')
    print('using trainfile:{}'.format(train_protofile))
    print('using  testfile:{}'.format(test_protofile))
    print('using deployfile:{}'.format(deploy_protofile))

    with open(train_protofile,'w') as f:
        train_net = net_builder(train_db,batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR,n_filters=n_filters,n_ip1=n_ip1)
        f.write(str(train_net))
        f.close
    with open(test_protofile,'w') as g:
        test_net = net_builder(test_db,batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR,n_filters=n_filters,n_ip1=n_ip1)
        g.write(str(test_net))
        g.close
    with open(deploy_protofile,'w') as h:
        deploy_net = net_builder(test_db,batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR,deploy=True)
        h.write(str(deploy_net))
        h.close

    solver = caffe.SGDSolver(proto_file_path)
    draw_net(deploy_protofile,os.path,join(nn_dir,'net_arch.jpg'))

    print('k,v all elements shape:')
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print('k, v[0] shape:')
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]
    solver.net.forward()  # train net
    solver.test_nets[0].forward()  # test net (there can be more than one)

    # we use a little trick to tile the first eight images
    if pc:
        pass

    #         plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray',block=False)
 #       plt.show(block=False)
    print solver.net.blobs['label'].data[:8]

    #%%time
    niter = 10000
    training_acc_threshold = 0.95
    test_interval = 100
    # losses will also be stored in the log
   # train_loss = zeros(niter)
   # train_acc = zeros(niter)
   # test_acc = zeros(int(np.ceil(niter / test_interval)))
   # train_acc2 = zeros(int(np.ceil(niter / test_interval)))
    train_loss = []
    train_acc = []
    test_acc = []
    train_acc2 = []
    running_avg_test_acc = 0
    previous_running_avg_test_acc = -1.0
    running_avg_upper_threshold = 1.001
    running_avg_lower_threshold = 0.999
    alpha = 0.1
    output = zeros((niter, 8, 10))
    train_size = lmdb_utils.db_size(train_db)
    test_size  = lmdb_utils.db_size(test_db)
    n_sample = test_size/batch_size
    print('db {} {} trainsize {} testsize {} batchsize {} n_samples {}'.format(train_db,test_db,train_size,test_size,batch_size,n_sample))
    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe

        solver.test_nets[0].forward(start='conv1')
#        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            #maybe this is whats sucking mem
            # store the train loss
            train_loss.append(solver.net.blobs['loss'].data)
    #        train_acc[it] = solver.net.blobs['accuracy'].data
            train_acc.append(solver.net.blobs['accuracy'].data)
            print('train loss {} train acc. {}'.format(train_loss[-1],train_acc[-1]))
#            print('train loss {} train acc. {}'.format(train_loss[it],train_acc[it]))
            # store the output on the first test batch
            # (start the forward pass at conv1 to avoid loading new data)
        #    train_acc2[it//test_interval] = solver.net.blobs['accuracy'].data
            train_acc2.append(solver.net.blobs['accuracy'].data)

            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(n_sample):
                solver.test_nets[0].forward()
                    #note the blob you check here has to be the final 'output layer'
                correct += sum(solver.test_nets[0].blobs['output_layer'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)

            print('{}. outputlayer.data {}  correct:{}'.format(test_it,solver.test_nets[0].blobs['output_layer'].data, solver.test_nets[0].blobs['label'].data))
#
            percent_correct = float(correct)/(n_sample*batch_size)
            print('correct {} n {} batchsize {} acc {} size(solver.test_nets[0].blob[output_layer]'.format(correct,n_sample,batch_size, percent_correct,len(solver.test_nets[0].blobs['label'].data)))
#            test_acc[it // test_interval] = percent_correct
            test_acc.append(percent_correct)
            running_avg_test_acc = (1-alpha)*running_avg_test_acc + alpha*test_acc[it//test_interval]
            print('acc so far:'+str(test_acc)+' running avg:'+str(running_avg_test_acc)+ ' previous:'+str(previous_running_avg_test_acc))
            drunning_avg = running_avg_test_acc/previous_running_avg_test_acc
            previous_running_avg_test_acc=running_avg_test_acc
#            if test_acc [it // test_interval] > training_acc_threshold:
            if test_acc [-1] > training_acc_threshold and 0:
                print('acc of {} is above required threshold of {}, thus stopping:'.format(test_acc,training_acc_threshold))
                break
            if drunning_avg > running_avg_lower_threshold and drunning_avg < running_avg_upper_threshold and 0:
                print('drunning avg of {} is between required thresholds of {} and {}, thus stopping:'.format(drunning_avg,running_avg_lower_threshold,running_avg_upper_threshold))
                break

        #figure 1 - train loss and train acc. for all forward passes
        plt.close("all")

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
    #    print('it {} trainloss {} len {}'.format(it,train_loss,len(train_loss)))
        l = len(train_loss)
 #       print('l {} train_loss {}'.format(l,train_loss))
        ax1.plot(arange(l), train_loss,'r.-')
        plt.yscale('log')
        ax1.set_title('train loss / accuracy for '+str(train_db))
        ax1.set_ylabel('train loss',color='r')
        ax1.set_xlabel('iteration',color='g')

        axb = ax1.twinx()
        l = len(train_acc)
 #       print('l {} train_acc {}'.format(l,train_acc))
        axb.plot(arange(l), train_acc,'b.-',label='train_acc')
#        plt.yscale('log')   #ValueError: Data has no positive values, and therefore can not be log-scaled.
        axb.set_ylabel('train acc.', color='b')
        legend = ax1.legend(loc='upper center', shadow=True)
        plt.show()

        #figure 2 - train and test acc every N passes
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        l = len(test_acc)
   #     print('l {} test_acc {}'.format(l,test_acc))
#        ax2.plot(arange(1+int(np.ceil(it / test_interval))), test_acc,'b.-',label='test_acc')
        ax2.plot(arange(l), test_acc,'b.-',label='test_acc')
        ax2.plot(arange(l), train_acc,'g.-',label='train_acc' )  #theres a mistake here, const value shown
        ax2.set_xlabel('iteration/'+str(test_interval))
        ax2.set_ylabel('test/train accuracy')
        ax2.set_title('train, test acc for '+str(train_db)+','+str(test_db))
        legend = ax2.legend(loc='upper center', shadow=True)
        #axes = plt.gca()
        #ax1.set_xlim([xmin,xmax])
        ax2.set_ylim([0,1])
        legend = ax2.legend(loc='upper center', shadow=True)
        plt.show()

    figname = os.path.join(nn_dir,'loss_and_testacc.png')
    fig1.savefig(figname)
    figname = os.path.join(nn_dir,'trainacc_and_testacc.png')
    fig2.savefig(figname)


    print('loss:'+str(train_loss))
    print('acc:'+str(test_acc))
    outfilename = os.path.join(nn_dir,'results.txt')
    with open(outfilename,'a') as f:
        f.write('dir {}\n db {}\nAccuracy\n'.format(nn_dir,train_db,test_db))
        f.write(str(test_acc))
#        f.write(str(train_net))
        f.close()

def inspect_net(caffemodel):
    net_param = caffe_pb2.NetParameter()
    net_str = open(caffemodel, 'r').read()
    net_param.ParseFromString(net_str)
    for l in net_param.layer:
        print net_param.layer[l].name  # first layer

def correct_deconv(proto):
    outlines = []
    in_deconv = False
    lines = proto.split('\n')
    outstring = ''
    for line in lines:
#        print('in  line:'+ line+str(in_deconv))
        if 'name' in line:
            if 'deconv' in line:
                in_deconv = True
            else:
                in_deconv = False
        if '}' in line:
            in_deconv = False
        if in_deconv and 'type:' in line and 'Convolution' in line:
            line = '  type:\"Deconvolution\"'
#        print('out line:'+ line)
        outlines.append(line)
        outstring = outstring+line+'\n\n'
    return outstring

def replace_pythonlayer(proto):
    pythonlayer = 'layer {\n    name: \"data\"\n    type: \"Python\"\n    top: \"data\"\n    top: \"label\"\n    python_param {\n    module: \"jrlayers\"\n    layer: \"JrPixlevel\"\n    param_str: \"{\\\"images_and_labels_file\\\": \\\"/home/jeremy/image_dbs/colorful_fashion_parsing_data/images_and_labelsfile_train.txt\\\", \\\"mean\\\": (104.0, 116.7, 122.7),\\\"augment\\\":True,\\\"augment_crop_size\\\":(224,224), \\\"batch_size\\\":9 }\"\n    }\n  }\n'
#    print pythonlayer
    in_data = False
    lines = proto.split('\n')
    outstring = ''
    new_layer_flag = False
    layer_buf = 'layer {\n'
    first_layer = True
    for i in range(len(lines)):
        line = lines[i]
#        print('in  line:'+ line+str(in_deconv))
        if 'layer {' in line or 'layer{' in line:
            start_layer = i #
            in_data = False
            new_layer_flag = True
        else:
            new_layer_flag = False
            if not in_data:
                layer_buf = layer_buf + line + '\n'
        if 'type' in line:
            if 'Data' in line:
                print('swapping in pythonlayer for datalayer')
                layer_buf = pythonlayer
                in_data = True
            else:
                in_data = False
        if new_layer_flag and not first_layer:
            print('layer buf:')
            print layer_buf
            first_layer = False
            outstring = outstring + layer_buf
            layer_buf = 'layer {\n'
        if new_layer_flag and first_layer:
            first_layer = False
    return outstring

#    param_str: "{\'images_and_labels_file\': \'/home/jeremy/image_dbs/colorful_fashion_parsing_data/images_and_labelsfile_train.txt\', \'mean\': (104.0, 116.7, 122.7),\'augment\':True,\'augment_crop_size\':(224,224), \'batch_size\':9 }"

'''
  convolution_param {
    num_output: 21
    bias_term: false
    kernel_size: 16
    stride: 8
  }
'''
if __name__ == "__main__":
#    run_net(googLeNet_2_inceptions,nn_dir,db_name+'_train',db_name+'_test',batch_size = batch_size,n_classes=11,meanB=B,meanR=R,meanG=G,n_filters=50,n_ip1=1000)
#    run_net(alexnet_linearized,nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=50,n_ip1=1000)

    proto = vgg16('thedb')
    proto = unet('thedb')
    proto = correct_deconv(str(proto))
    proto = replace_pythonlayer(proto)


    with open('train_experiment.prototxt','w') as f:
        f.write(str(proto))
        f.close()
