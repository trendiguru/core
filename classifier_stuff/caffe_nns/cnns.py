# coding: utf-8
__author__ = 'jeremy'
import sys
import os
import socket
from pylab import *
from trendi.classifier_stuff.caffe_nns import lmdb_utils
import sys
import caffe
import cv2
from trendi.utils import imutils
#import Image
#import matplotlib
from trendi import Utils


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

#from matplotlib import *
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

#get_ipython().system(u'data/mnist/get_mnist.sh')
#get_ipython().system(u'examples/mnist/create_mnist.sh')

import lmdb
from PIL import Image


#label = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_label', transform_param=dict(scale=1./255), ntop=1)
#data = L.Data(batch_size=99, backend=P.Data.LMDB, source='train_data', transform_param=dict(scale=1./255), ntop=1)

'''
  layers {
name: "images"
type: IMAGE_DATA
top: "data"
top: "label"
image_data_param {
source: "custom_train_lmdb"
batch_size: 4
crop_size: 227
shuffle: true
}


  layer {
    name: "data"
    type: "ImageData"
    top: "data"
    top: "label"
    transform_param {
      mirror: false
      crop_size: 227
      mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
    }
    image_data_param {
      source: "examples/_temp/file_list.txt"
      batch_size: 50
      new_height: 256
      new_width: 256
    }
  }
'''


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
    # snarpshot - snapshot intermediate results
    # snarpshot prefix - dir for snapshot  - maybe requires '/' at end?
    # solver_mode - CPU or GPU
    prototxt ={ 'train_net':train_file,
                        'test_net': test_file,
                        'test_iter': test_iter,
                        'test_interval': 500,
                        'base_lr': 0.01,
                        'momentum': 0.9,
                        'weight_decay': 0.0005,
                        'lr_policy': "inv",
                        'gamma': 0.0001,
                        'power': 0.75,
                        'display': 10,
                        'max_iter': 10000,
                        'snapshot': 1000,
                        'snapshot_prefix': dir+'/net',
                        'solver_mode':solver_mode }

    print('writing prototxt:'+str(prototxt))
    with open(proto_filename,'w') as f:
        for key, val in prototxt.iteritems():
            line=key+':'+str(val)+'\n'
            if isinstance(val,basestring) and key is not 'solver_mode':
                line=key+':\"'+str(val)+'\"\n'
            f.write(line)

#net: "examples/mnist/lenet_train_test.prototxt"
#test_iter: 100
#test_interval: 500
#base_lr: 0.01
#momentum: 0.9
#weight_decay: 0.0005
#lr_policy: "inv"
#gamma: 0.0001
#power: 0.75
#display: 100
#max_iter: 10000
#snapshot: 5000
#snapshot_prefix: "examples/mnist/lenet"
#solver_mode: GPU

#googlenet:
'''net: "models/bvlc_googlenet/googLeNet.prototxt"
test_iter: 1000
test_interval: 4000
test_initialization: false
display: 40
average_loss: 40
base_lr: 0.01
lr_policy: "poly"
power: 0.5
max_iter: 2400000
momentum: 0.9
weight_decay: 0.0002
snapshot: 40000
snapshot_prefix: "models/bvlc_googlenet/bvlc_googlenet_quick"
solver_mode: GPU
'''

def lenet(lmdb, batch_size):  #test_iter * batch_size <= n_samples!!!
    lr_mult1 = 1
    lr_mult2 = 2
    n=caffe.NetSpec()
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,transform_param=dict(scale=1./255),ntop=2)

    n.conv1 = L.Convolution(n.data,  kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
#    L.Convolution(bottom, param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
 #       kernel_h=kh, kernel_w=kw, stride=stride, num_output=nout, pad=pad,
  #      weight_filler=dict(type='gaussian', std=0.1, sparse=sparse),
   #     bias_filler=dict(type='constant', value=0))

    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride=1,num_output=20,weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

#    n.ip1 = L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.ip1 = L.InnerProduct(n.pool2,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=500,weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=10,weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2,n.label)
    return n.to_proto()
#missing from conv and ip1,2:
    # param {
#    lr_mult: 1
#  }
 # param {
  #  lr_mult: 2
  #}
'''
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
}

bvlccaffenet conv layer
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  # learning rate and decay multipliers for the filters
  param { lr_mult: 1 decay_mult: 1 }
  # learning rate and decay multipliers for the biases
  param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 96     # learn 96 filters
    kernel_size: 11    # each filter is 11x11
    stride: 4          # step 4 pixels between each filter application
    weight_filler {
      type: "gaussian" # initialize the filters from a Gaussian
      std: 0.01        # distribution with stdev 0.01 (default mean: 0)
    }
    bias_filler {
      type: "constant" # initialize the biases to zero (0)
      value: 0
    }
  }
}
'''

'''       from nate's vgg
  1. slicing the layers is by a '_'
    2. slicing each layer data is by a 'x'
    3. 'C2D' = Sequential.add(Convolution2D())
        3.1 number of kernels (int)
        3.2 length of kernel along dim 1(int)
        3.3 length of kernel along dim 2(int)
        3.4 border_mode -> 'valid' / 'same'
        3.5 input_shape -> 3x dimentions (int)
    4. 'MP' = max pooling layer
        4.1 pool length dim 1 (int)
        4.2 pool length dim 2 (int)
    5. 'AP' = average pooling layer
        5.1 pool length dim 1 (int)
        5.2 pool length dim 2 (int)
    4. 'F' = Sequential.add(Flatten())
    5. 'A' = Sequential.add(Activation(->)) : 'relu' / 'sigmoid' / 'hard_sigmoid / 'softmax' / 'tanh' / 'softplus' / 'linear'
    6. 'DO' = Sequential.add(Dropout(->)) : 0.0 <= value <= 1.0
    7. 'D' = Sequential.add(Dense(->)) : int value > 0 (fu
'''
def vggnet(db, batch_size,n_classes=11,meanB=128,meanG=128,meanR=128,n_filters=50,n_ip1=1000):
    print('running mynet n {} B {} G {} R {} db {} batchsize {}'.format(n_classes,meanB,meanG,meanR,db,batch_size))
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0

    n=caffe.NetSpec()
    if meanB is not None and meanG is not None and meanR is not None:
        print('using vector mean ({} {} {})'.format(meanB,meanG,meanR ))
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=[meanB,meanG,meanR]),ntop=2)
    elif meanB:
        print('using 1D mean {} '.format(meanB))
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=meanB),ntop=2)
    else:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255),ntop=2)

    n.conv1 = L.Convolution(n.data,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=5,stride = 1, num_output=n_filters,weight_filler=dict(type='xavier'))

#is relu required after every conv?
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)

#    L.Convolution(bottom, param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
 #       kernel_h=kh, kernel_w=kw, stride=stride, num_output=nout, pad=pad,
  #      weight_filler=dict(type='gaussian', std=0.1, sparse=sparse),
   #     bias_filler=dict(type='constant', value=0))

    n.ip1 = L.InnerProduct(n.pool1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_ip1,weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.ip1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_classes,weight_filler=dict(type='xavier'))
    n.accuracy = L.Accuracy(n.ip2,n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2,n.label)
    return n.to_proto()

def alexnet_linearized(db, batch_size,n_classes=11,meanB=128,meanG=128,meanR=128,n_filters=50,n_ip1=1000):
    print('building alexnet n {} B {} G {} R {} db {} batchsize {}'.format(n_classes,meanB,meanG,meanR,db,batch_size))
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0

    n=caffe.NetSpec()
    if meanB is not None and meanG is not None and meanR is not None:
        print('using vector mean ({} {} {})'.format(meanB,meanG,meanR ))
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=[meanB,meanG,meanR]),ntop=2)
    elif meanB:
        print('using 1D mean {} '.format(meanB))
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=meanB),ntop=2)
    else:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255),ntop=2)

    n.conv1 = L.Convolution(n.data,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=10,stride = 4, num_output=48,weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=1, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=5,stride = 1, num_output=128,weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=1, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.pool2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=3,stride = 1, num_output=192,weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.conv3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=3,stride = 1, num_output=192,weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.conv4,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=3,stride = 1, num_output=128,weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    n.pool3 = L.Pooling(n.conv5, kernel_size=3, stride=1, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool3,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=2048,weight_filler=dict(type='xavier'))
    n.relu6 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.ip1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=2048,weight_filler=dict(type='xavier'))
    n.relu7 = L.ReLU(n.ip2, in_place=True)
    n.output_layer = L.InnerProduct(n.ip2,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_classes,weight_filler=dict(type='xavier'))
#maybe add relu here?
    n.accuracy = L.Accuracy(n.output_layer,n.label)
    n.loss = L.SoftmaxWithLoss(n.output_layer,n.label)
    return n.to_proto()

def run_lenet():
    if host == 'jr-ThinkPad-X1-Carbon':
        pc = True
        os.chdir('/home/jr/sw/caffe')
    else:
        pc = False
        os.chdir('/root/caffe')
    with open('examples/mnist/lenet_auto_train.prototxt','w') as f:
        f.write(str(lenet('examples/mnist/mnist_train_lmdb',64)))
    with open('examples/mnist/lenet_auto_test.prototxt','w') as f:
        f.write(str(lenet('examples/mnist/mnist_test_lmdb',100)))
    host = socket.gethostname()
    print('host:'+str(host))
    if pc:
        print('using cpu only on '+str(host))
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)

    solver = caffe.SGDSolver('examples/mnist/lenet_auto_solver.prototxt')
    [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    [(k, v[0].data.shape) for k, v in solver.net.params.items()]
    solver.net.forward()  # train net
    solver.test_nets[0].forward()  # test net (there can be more than one)
    # we use a little trick to tile the first eight images
    if pc:
        plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray',block=False)
 #       plt.show(block=False)
    print solver.net.blobs['label'].data[:8]
    if pc:
        plt.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
#        plt.draw()
    print solver.test_nets[0].blobs['label'].data[:8]
    solver.step(1)
    if pc:
        plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
           .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray')
 #       plt.draw()

    #%%time
    niter = 200
    test_interval = 25
    # losses will also be stored in the log
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter / test_interval)))
    output = zeros((niter, 8, 10))

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
            test_acc[it // test_interval] = correct / 1e4
    if pc:
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(arange(niter), train_loss)
        ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('test accuracy')
        plt.show()

def mynet(db, batch_size,n_classes=11,meanB=128,meanG=128,meanR=128,n_filters=50,n_ip1=1000):
    print('running mynet n {} B {} G {} R {} db {} batchsize {}'.format(n_classes,meanB,meanG,meanR,db,batch_size))
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0

    n=caffe.NetSpec()
    if meanB is not None and meanG is not None and meanR is not None:
        print('using vector mean ({} {} {})'.format(meanB,meanG,meanR ))
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=[meanB,meanG,meanR]),ntop=2)
    elif meanB:
        print('using 1D mean {} '.format(meanB))
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=meanB),ntop=2)
    else:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255),ntop=2)

    n.conv1 = L.Convolution(n.data,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=5,stride = 1, num_output=n_filters,weight_filler=dict(type='xavier'))

#is relu required after every conv?
    n.relu1 = L.ReLU(n.conv1, in_place=True)

#    L.Convolution(bottom, param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
 #       kernel_h=kh, kernel_w=kw, stride=stride, num_output=nout, pad=pad,
  #      weight_filler=dict(type='gaussian', std=0.1, sparse=sparse),
   #     bias_filler=dict(type='constant', value=0))

    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2 = L.Convolution(n.pool1,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                                          dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=5,stride = 1, num_output=n_filters,weight_filler=dict(type='xavier'))

    n.relu2 = L.ReLU(n.conv2, in_place=True)
#    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv3 = L.Convolution(n.conv2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            kernel_size=5,stride = 1, num_output=n_filters,weight_filler=dict(type='xavier'))

    n.relu3 = L.ReLU(n.conv3, in_place=True)
  #  n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)


    n.conv4 = L.Convolution(n.conv3,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride=1,num_output=n_filters,weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
   # n.pool4 = L.Pooling(n.conv4, kernel_size=2, stride=2, pool=P.Pooling.MAX)

#    n.ip1 = L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.ip1 = L.InnerProduct(n.conv4,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_ip1,weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.ip1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_classes,weight_filler=dict(type='xavier'))
    n.accuracy = L.Accuracy(n.ip2,n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2,n.label)
    return n.to_proto()


def googLeNet(db, batch_size, n_classes=11, meanB=128, meanG=128, meanR=128):
#    print('running mynet n {} B {} G {} R {] db {} batchsize {} '.format(n_classes,meanB,meanG,meanR,db,batch_size))
    print('running mynet n {}  batchsize {} '.format(n_classes,batch_size))
   #crop size 224
    lr_mult1 = 1
    lr_mult2 = 2
    decay_mult1 =1
    decay_mult2 =0

    n=caffe.NetSpec()
    if meanB is not None and meanG is not None and meanR is not None:
        print('using vector mean ({} {} {})'.format(meanB,meanG,meanR ))
        n.data_1,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=[meanB,meanG,meanR],mirror=True),ntop=2)
    elif meanB:
        print('using 1D mean {} '.format(meanB))
        n.data_1,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=meanB,mirror=True),ntop=2)
    else:
        n.data_1,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mirror=True),ntop=2)

    n.conv1_7x7_s2_3= L.Convolution(n.data_1,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=64,
                            pad = 3,
                            kernel_size=7,
                            stride = 2,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.conv1_relu_7x7_4 = L.ReLU(n.conv1_7x7_s2_3,in_place=True)
    n.pool1_3x3_s2_5 = L.Pooling(n.conv1_7x7_s2_3, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.pool1_norm1_6 = L.LRN(n.pool1_3x3_s2_5,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))

    n.conv2_3x3_reduce_7 = L.Convolution(n.pool1_norm1_6,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=64,
                            kernel_size=1,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.conv2_relu_3x3_reduce_8 = L.ReLU(n.conv2_3x3reduce_7,in_place=True)

    n.conv2_3x3_9 = L.Convolution(n.conv2_3x3reduce_7,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=192,
                            pad = 1,
                            kernel_size=3,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.conv2_relu_3x3_10 = L.ReLU(n.conv2_3x3_9,in_place=True)
    n.conv2_norm2_11 = L.LRN(n.conv2_3x3_9,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))
    n.pool2_3x3_s2_12 = L.Pooling(n.conv2_norm2_11, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    n.inception_3a_1x1_13 = L.Convolution(n.pool2_3x3s2_12,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=64,
                            kernel_size=1,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.inception_3a_relu1x1_14 = L.ReLU(n.inception_3a_1x1_13,in_place=True)
# does inception_3a get used later
    n.inception_3a_3x3_reduce_15 =  L.Convolution(n.pool2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=96,
                            kernel_size=1,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.inception_3a_relu_3x3_reduce_16 = L.ReLU(n.inception_3a_3x3_reduce_15,in_place=True)
    n.inception_3a_3x3_17 =  L.Convolution(n.inception_3a_3x3_reduce_15,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=128,
                            pad = 1,
                            kernel_size=3,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.inception_3a_relu3x3_18 = L.ReLU(n.inception_3a_3x3_17,in_place=True)
    n.inception_3a_5x5_reduce_19 =  L.Convolution(n.pool2_3x3s2_12,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=16,
                            pad = 1,
                            kernel_size=1,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.inception_3a_relu5x5_reduce_20 = L.ReLU(n.inception_3a_5x5_reduce_19,in_place=True)
    n.inception_3a_5x5_21 =  L.Convolution(n.inception_3a_relu5x5_reduce_20,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=32,
                            pad = 2,
                            kernel_size=5,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.inception_3a_relu5x5_22 = L.ReLU(n.inception_3a_5x5_21,in_place=True)
    n.inception_3a_pool_23 = L.Pooling(n.pool2_3x3_s2_12, kernel_size=3, stride=1,pad=1, pool=P.Pooling.MAX)
    n.inception_3a_pool_proj_24 =  L.Convolution(n.inception_3a_pool_23,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
                            dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=32,
                            weight_filler=dict(type='xavier'),
                            bias_filler=dict(type='constant',value=0.2))
    n.inception_3a_relu_pool_proj_25 = L.ReLU(n.inception_3a_pool_proj_24, in_place=True)
    n.inception_3a_output_26 = L.Concat(bottom=[n.inception_3a_1x1_13,n.inception_3a_3x3_17,n.inception_3a_5x5_21,n.inception_3a_pool_proj_24])

#    n.lrn1 = L.LRN(n.pool1,lrn_param=dict(local_size=5,alpha=0.0001,beta=0.75))


#    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

#    n.conv3 = L.Convolution(n.pool2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),
  #                                        dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
    #                        kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))

#    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)


#    n.conv4 = L.Convolution(n.pool3,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
  #                          kernel_size=5,stride=1,num_output=50,weight_filler=dict(type='xavier'))
   # n.pool4 = L.Pooling(n.conv4, kernel_size=2, stride=2, pool=P.Pooling.MAX)

#    n.ip1 = L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
#    n.ip1 = L.InnerProduct(n.pool3,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=1000,weight_filler=dict(type='xavier'))
  #  n.relu1 = L.ReLU(n.ip1, in_place=True)
   # n.ip2 = L.InnerProduct(n.relu1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_classes,weight_filler=dict(type='xavier'))
    #n.accuracy = L.Accuracy(n.ip2,n.label)
    #n.loss = L.SoftmaxWithLoss(n.ip2,n.label)
    return n.to_proto()


'''to conditionally include, use
  include {
    phase: TRAIN
  }

(data)
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 104
    mean_value: 117
    mean_value: 123
  }
  data_param {
    source: "examples/imagenet/ilsvrc12_train_lmdb"
    batch_size: 32
    backend: LMDB
  }


(at end)
  layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }

}

'''

#missing from conv and ip1,2:
    # param {
#    lr_mult: 1
#  }
 # param {
  #  lr_mult: 2
  #}

def run_net(net_builder,nn_dir,train_db,test_db,batch_size = 64,n_classes=11,meanB=None,meanG=None,meanR=None,n_filters=50,n_ip1=1000):
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
    print('using trainfile:{}'.format(train_protofile))
    print('using  testfile:{}'.format(test_protofile))

    with open(train_protofile,'w') as f:
        train_net = net_builder(train_db,batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR,n_filters=n_filters,n_ip1=n_ip1)
        f.write(str(train_net))
        f.close
    with open(test_protofile,'w') as g:
        test_net = net_builder(test_db,batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR,n_filters=n_filters,n_ip1=n_ip1)
        g.write(str(test_net))
        g.close

    solver = caffe.SGDSolver(proto_file_path)

#    caffe.draw_net_to_file(proto_file_path,,'net_topo.png', rankdir='LR')
  #  caffe.proto.caffe_pb2.NetParameter protocol buffer.
   #         datum = caffe.proto.caffe_pb2.Datum()
    netparam = caffe.proto.caffe_pb2.NetParameter()

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
    test_interval = 100
    # losses will also be stored in the log
    train_loss = zeros(niter)
    train_acc = zeros(niter)
    test_acc = zeros(int(np.ceil(niter / test_interval)))
    output = zeros((niter, 8, 10))

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        train_acc[it] = solver.net.blobs['accuracy'].data
        print('train loss {} train acc. {}'.format(train_loss[it],train_acc[it]))
        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
#        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        n_sample = 100
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(n_sample):
                solver.test_nets[0].forward()
                    #note the blob you check here has to be the final 'output layer'
                correct += sum(solver.test_nets[0].blobs['output_layer'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
            percent_correct = float(correct)/(n_sample*batch_size)
            print('correct {} n {} batchsize {} acc {} size(solver.test_nets[0].blob[output_layer]'.format(correct,n_sample,batch_size, percent_correct,len(solver.test_nets[0].blobs['label'].data)))
            test_acc[it // test_interval] = percent_correct
            print('acc so far:'+str(test_acc))
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), train_loss)
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    fig = plt.figure()
    fig.savefig('out.png')
    if pc:
        plt.show()
    print('loss:'+str(train_loss))
    print('acc:'+str(test_acc))
    outfilename = os.path.join(nn_dir,'results.txt')
    with open(outfilename,'a') as f:
        f.write('dir {} db {}'.format(nn_dir,train_db,test_db))
        f.write(str(test_acc))
        f.write(str(train_net))
        f.close()



host = socket.gethostname()
print('host:'+str(host))

if __name__ == "__main__":
    if host == 'jr-ThinkPad-X1-Carbon':
        pc = True
        dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/dataset'
        dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/plusminus_data'
        nn_dir = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/plusminus_alextest'
        max_images_per_class = 100
        solver_mode = 'CPU'
        B=142
        G=151
        R=162
        db_name = 'pluszero'
        db_name = 'mydb200'
        db_name = 'plus_zero'
    else:
        base_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/'
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/plusminus_data'  #b2
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/populated_items'  #b2
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset/cropped'  #b2
        nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/alexnet7'  #b2
#        nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/vgg_'  #b2
        max_images_per_class = 15000
        pc = False
        solver_mode = 'GPU'
        B=112
        G=123
        R=136
        db_name = 'highly_populated_cropped'

#    h,w,d,B,G,R,n = imutils.image_stats_from_dir_of_ditestrs(dir_of_dirs)
    resize_x = None
    resize_y=None
    print('dir:'+dir_of_dirs)
    print('rgb:'+str(R)+','+str(G)+','+str(B))
    print('mode:'+solver_mode)




#    lmdb_utils.kill_db(db_name)
    test_iter = 100
    batch_size = 32  #use powers of 2 for better perf (supposedly)
# out of mem possibly correctable:    Reading dangerously large protocol message.  If the message turns out to be larger than 2147483647 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h

    find_averages = False
    if find_averages:
        retval = imutils.image_stats_from_dir_of_dirs(dir_of_dirs)
#            return([avg_h,avg_w,avg_d,avg_B,avg_G,avg_R,totfiles])
        B = retval[3]
        G = retval[4]
        R = retval[5]

#    lmdb_utils.inspect_db(db_name+'.train')
  #  lmdb_utils.inspect_db(db_name+'.test')
    generate_db = False
    if generate_db:
        n_test_classes,test_populations,image_number_test = lmdb_utils.interleaved_dir_of_dirs_to_lmdb(db_name,dir_of_dirs,
                                                                                           max_images_per_class = 10000,test_or_train='test',resize_x=resize_x,resize_y=resize_y,
                                                                                        use_visual_output=False,n_channels=3)
        print('testclasses {} populations {} tot_images {} '.format(n_test_classes,test_populations,image_number_test))

        n_train_classes,train_populations,image_number_train = lmdb_utils.interleaved_dir_of_dirs_to_lmdb(db_name,dir_of_dirs,
                                                                                              max_images_per_class =10000,test_or_train='train',resize_x=resize_x,resize_y=resize_y,
                                                                                        use_visual_output=False,n_channels=3)
        tot_train_samples = np.sum(train_populations)
        tot_test_samples = np.sum(test_populations)
        n_classes = n_test_classes
        print('testclasses {} populations {} tot_images {} '.format(n_test_classes,test_populations,image_number_test))
        print('trainclasses {} populations {} tot_images {} '.format(n_train_classes,train_populations,image_number_train))
        print('sum test pops {}  sum train pops {}  testiter {} batch_size {}'.format(tot_train_samples,tot_test_samples,test_iter,batch_size))
    else:
        n_classes  = 4

#    lmdb_utils.inspect_db(db_name+'.train')
  #  lmdb_utils.inspect_db(db_name+'.test')
   # raw_input('enter to cont')
#    nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/conv_relu_x4_nopool_50filters'  #b2
  #  run_my_net(nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=50,n_ip1=1000)
   # nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/conv_relu_x4_nopool_70filters'  #b2
  #  run_my_net(nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=70,n_ip1=1000)

#out of memory!!
    #      nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/conv_relu_x4_nopool_100filters'  #b2
    #run_my_net(nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=100,n_ip1=1000)
#    nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/conv_relu_x4_nopool_200filters'  #b2
  #  run_my_net(nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=200,n_ip1=1000)
   # nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/conv_relu_x4_nopool_200filters_2000ip1'  #b2
    #run_my_net(nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=200,n_ip1=2000)

#out of memory!!
#    nn_dir = '/home/jeremy/core/classifier_stuff/caffe_nns/conv_relu_x4_nopool_70filters_2000ip1'  #b2
    #     run_my_net(nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=70,n_ip1=2000)

    run_net(alexnet_linearized,nn_dir,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G,n_filters=50,n_ip1=1000)


#to train at cli:
# caffe train -solver=solver_prototxt
#to test
#caffe test -mode test.prototxt -weights model.caffemodel -gpu 0 -iterations 100
#/opt/caffe/build/tools/caffe test -model cropped_dataset/my_solver.test.prototxt -weights cropped_dataset_iter_3000.caffemodel  -gpu 0 -iterations 500