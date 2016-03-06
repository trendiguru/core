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
#import Image
#import matplotlib


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
    # test_iter specifies how many forward passes the test should carry out.
    # In the case of MNIST, we have test batch size 100 and 100 test iterations,
    # covering the full 10,000 testing images.
    # test_interval - Carry out testing every 500 training iterations.
    # base_lr - The base learning rate, momentum and the weight decay of the network.
    # lr_policy - The learning rate policy
    # display - Display every n iterations
    # max_iter - The maximum number of iterations
    # snarpshot - snapshot intermediate results
    prototxt ={ 'train_net':train_file,
                        'test_net': test_file,
                        'test_iter': test_iter,
                        'test_interval': 200,
                        'base_lr': 0.01,
                        'momentum': 0.9,
                        'weight_decay': 0.0005,
                        'lr_policy': "inv",
                        'gamma': 0.0001,
                        'power': 0.75,
                        'display': 100,
                        'max_iter': 10000,
                        'snapshot': 5000,
                        'snapshot_prefix': dir,
                        'solver_mode':solver_mode }

    print prototxt
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


def lenet(lmdb, batch_size):  #test_iter * batch_size <= n_samples!!!
    lr_mult1 = 1
    lr_mult2 = 2
    n=caffe.NetSpec()
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,transform_param=dict(scale=1./255),ntop=2)

#    n.conv1 = L.Convolution(n.data,kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
    n.conv1 = L.Convolution(n.data,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
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

'''



def run_lenet():
    host = socket.gethostname()
    print('host:'+str(host))
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

def mynet(db, batch_size,n_classes=11,meanB=128,meanG=128,meanR=128  ):
    lr_mult1 = 1
    lr_mult2 = 2
    n=caffe.NetSpec()
    if meanB:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=meanB),ntop=2)
    else:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255),ntop=2)

#    n.conv1 = L.Convolution(n.data,kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
    n.conv1 = L.Convolution(n.data,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
#    L.Convolution(bottom, param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
 #       kernel_h=kh, kernel_w=kw, stride=stride, num_output=nout, pad=pad,
  #      weight_filler=dict(type='gaussian', std=0.1, sparse=sparse),
   #     bias_filler=dict(type='constant', value=0))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2 = L.Convolution(n.pool1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride=1,num_output=50,weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv3 = L.Convolution(n.pool2,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride = 1, num_output=50,weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv4 = L.Convolution(n.pool3,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],
                            kernel_size=5,stride=1,num_output=20,weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv4, kernel_size=2, stride=2, pool=P.Pooling.MAX)

#    n.ip1 = L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.ip1 = L.InnerProduct(n.pool4,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=500,weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],num_output=n_classes,weight_filler=dict(type='xavier'))
#    n.accuracy = L.Accuracy(n.ip2,n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2,n.label)
    return n.to_proto()

'''to conditionally include, use
  include {
    phase: TRAIN
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

def run_my_net(nn_dir,train_db,test_db,batch_size = 64,n_classes=11,meanB=None,meanG=None,meanR=None):
    host = socket.gethostname()
    print('host:'+str(host))
    if host == 'jr-ThinkPad-X1-Carbon':
        pc = True
        solver_mode = 'CPU'
        caffe.set_mode_cpu()
    else:
        pc = False
        solver_mode = 'GPU'
        caffe.set_mode_gpu()
        caffe.set_device(0)

    proto_filename = 'my_solver.prototxt'
    proto_file_path = os.path.join(nn_dir,'my_solver.prototxt')
    test_iter = 100
    write_prototxt(proto_file_path,test_iter = test_iter,solver_mode=solver_mode)
    proto_file_base = proto_filename.split('prototxt')[0]
    train_protofile = os.path.join(nn_dir,proto_file_base+'train.prototxt')
    test_protofile = os.path.join(nn_dir,proto_file_base+'test.prototxt')
    print('using trainfile:{} testfile:{}'.format(train_protofile,test_protofile))

    with open(train_protofile,'w') as f:
        f.write(str(mynet(train_db,batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR)))
        f.close
    with open(test_protofile,'w') as g:
        g.write(str(mynet(test_db, batch_size = batch_size,n_classes=n_classes,meanB=meanB,meanG=meanG,meanR=meanR)))
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
                correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
            percent_correct = float(correct)/(n_sample*batch_size)
            print('correct {} n {} batchsize {} %{}:'.format(correct,n_sample,len(solver.test_nets[0].blobs['label'].data), percent_correct))
            test_acc[it // test_interval] = percent_correct

    if pc:
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(arange(niter), train_loss)
        ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('test accuracy')
        plt.show()
    print('loss:'+str(train_loss))
    print('acc:'+str(test_acc))

def load_net(prototxt,caffemodel,mean_B=128,mean_G=128,mean_R=128,image='../../images/female1.jpg',image_width=128,image_height=128,image_depth=3,batch_size=256):
    host = socket.gethostname()
    print('host:'+str(host))
    pc = False
    caffe.set_mode_gpu()
    if host == 'jr-ThinkPad-X1-Carbon':
        pc = True
        caffe.set_mode_cpu()
    net = caffe.Net(prototxt,caffemodel,caffe.TEST)
    # see http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
#    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
 #   mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    mu = [mean_B,mean_G,mean_R]
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
#    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(batch_size,        # batch size
                              image_depth,         # 3-channel (BGR) images
                             image_width, image_height)  # image size is 227x227
    image = caffe.io.load_image(image)
    transformed_image = transformer.preprocess('data', image)
#    plt.imshow(image)
# copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print 'predicted classes:', output_prob
    print 'predicted class is:', output_prob.argmax()


def test_net(prototxt,caffemodel, db_path):
    net = caffe.Net(prototxt, caffemodel,caffe.TEST)
    caffe.set_mode_cpu()
    lmdb_env = lmdb.open(db_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    count = 0
    correct = 0
    max_to_test = 100
    for key, value in lmdb_cursor:
        print "Count:"
        print count
        count = count + 1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        out = net.forward_all(data=np.asarray([image]))
        predicted_label = out['prob'][0].argmax(axis=0)
        if label == predicted_label[0][0]:
            correct = correct + 1
        print("Label is class " + str(label) + ", predicted class is " + str(predicted_label[0][0]))
        if count == max_to_test:
            break
    print(str(correct) + " out of " + str(count) + " were classified correctly")



if __name__ == "__main__":
    host = socket.gethostname()
    print('host:'+str(host))
    if host == 'jr-ThinkPad-X1-Carbon':
        pc = True
        dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/plusminus_data'
        max_images_per_class = 10000
        solver_mode = 'CPU'
    else:
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/plusminus_data'  #b2
        dir_of_dirs = '/home/jeremy/core/classifier_stuff/caffe_nns/dataset'  #b2
        max_images_per_class = 100000
        pc = False
        solver_mode = 'GPU'

    print('dir:'+dir_of_dirs)
#    h,w,d,B,G,R,n = imutils.image_stats_from_dir_of_ditestrs(dir_of_dirs)
    resize_x = 200
    #resize_y = int(h*128/w)
    resize_y=200
   # B=int(B)
   # G=int(G)
    #R=int(R)
    B=142
    G=151
    R=162
    db_name = 'pluszero'
    db_name = 'plus_zero'
    db_name = 'mydb200'
#    lmdb_utils.kill_db(db_name)
    test_iter = 100
    batch_size = 32  #use powers of 2 for better perf (supposedly)

    generate_db = True
    if generate_db:
        n_test_classes,test_populations,image_number_test = lmdb_utils.interleaved_dir_of_dirs_to_lmdb(db_name,dir_of_dirs,
                                                                                           max_images_per_class =max_images_per_class,test_or_train='test',resize_x=resize_x,resize_y=resize_y,
                                                                                        use_visual_output=False,n_channels=3)
        n_train_classes,train_populations,image_number_train = lmdb_utils.interleaved_dir_of_dirs_to_lmdb(db_name,dir_of_dirs,
                                                                                              max_images_per_class =max_images_per_class,test_or_train='train',resize_x=resize_x,resize_y=resize_y,
                                                                                        use_visual_output=False,n_channels=3)
        tot_train_samples = np.sum(train_populations)
        tot_test_samples = np.sum(test_populations)
        n_classes = n_test_classes
        print('trainclasses {} sum {} n {} test classes{} sum {} n {} testiter {} batch_size {}'.format(n_train_classes,tot_train_samples,image_number_train,n_test_classes,tot_test_samples,image_number_test,test_iter,batch_size))
    else:
        n_classes  = 2

    run_my_net(dir_of_dirs,db_name+'.train',db_name+'.test',batch_size = batch_size,n_classes=n_classes,meanB=B,meanR=R,meanG=G)
