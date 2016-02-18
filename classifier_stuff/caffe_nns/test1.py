# coding: utf-8
__author__ = 'jeremy'
import sys
import socket
from pylab import *
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

'''  layer {
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

def lenet(lmdb, batch_size):
    n=caffe.NetSpec()
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,transform_param=dict(scale=1./255),ntop=2)
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,transform_param=dict(scale=1./255),ntop=2)
    n.conv1 = L.Convolution(n.data,kernel_size=5,num_output=20,weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1,kernel_size=5,num_output=20,weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1,num_output=10,weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2,n.label)
    return n.to_proto()

def run_net():
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


################LMDB FUN (originally) RIPPED FROM http://deepdish.io/2015/04/28/creating-lmdb-in-python/
#############changes by awesome d.j. jazzy jer  awesomest hAckz0r evarr
def dir_of_dirs_to_lmdb(dbname,dir_of_dirs,test_or_train=None):
    only_dirs = [dir for dir in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,dir))]
    only_dirs.sort()
    print(str(len(only_dirs))+' dirs:'+str(only_dirs)+' in '+dir_of_dirs)

    map_size = 1e13  #size of db in bytes, can also be done by 10X actual size  as in:
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
#    map_size = X.nbytes * 10

    if test_or_train:
        dbname = dbname+test_or_train
    env = lmdb.open(dbname, map_size=map_size)
    classno = 0
    image_number =0
    with env.begin(write=True) as txn:
    # txn is a Transaction object
        for a_dir in only_dirs:
            # do only test or train dirs if this param was sent
            if (not test_or_train) or dir[0:4]==test_or_train[0:4]:
                fulldir = os.path.join(dir_of_dirs,a_dir)
                print('fulldir:'+str(fulldir))
                only_files = [f for f in os.listdir(fulldir) if os.path.isfile(os.path.join(fulldir, f))]
                n = len(only_files)
                print('n files {} in {}'.format(n,dir))
                for a_file in only_files:
                    fullname = os.path.join(fulldir,a_file)
                    #img_arr = mpimg.imread(fullname)  #if you don't have cv2 handy use matplotlib
                    img_arr = cv2.imread(fullname)
                    if img_arr is not None:
                        #    N = 1000
                        #    # Let's pretend this is interesting data
                        #    X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
                         #   y = np.zeros(N, dtype=np.int64)
                        cv2.imshow('img',img_arr)
                        cv2.waitKey(10)
                        datum = caffe.proto.caffe_pb2.Datum()
                        datum.channels = img_arr.shape[2]
                        datum.height = img_arr.shape[0]
                        datum.width = img_arr.shape[1]
                        datum.data = img_arr.tobytes()  # or .tostring() if numpy < 1.9
                        datum.label = classno
                        str_id = '{:08}'.format(image_number)
                        print('strid:'+str(str_id))
                        # The encode is only essential in Python 3
                        txn.put(str_id.encode('ascii'), datum.SerializeToString())
                        image_number += 1
                    else:
                        print('couldnt read '+a_file)
            classno += 1




    #You can also open up and inspect an existing LMDB database from Python:
def inspect_db():
    env = lmdb.open('mylmdb', readonly=True)
    with env.begin() as txn:
        raw_datum = txn.get(b'00000000')

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label

    #Iterating <key, value> pairs is also easy:

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            print(key, value)

def crude_lmdb():
    in_db = lmdb.open('image-lmdb', map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
 #           im = np.array(Image.open(in_)) # or load whatever ndarray you need
  #          im = im[:,:,::-1]
   #         im = im.transpose((2,0,1))
    #        im_dat = caffe.io.array_to_datum(im)
      #      in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()

if __name__ == "__main__":
    dir_of_dirs = '/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/only_train'
    print('dir:'+dir_of_dirs)
    dir_of_dirs_to_lmdb('testdb',dir_of_dirs,test_or_train='test')

#    test_or_training_textfile(dir_of_dirs,test_or_train='train')
#    Utils.remove_duplicate_files('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/pd_output_brain1/')
#    image_stats_from_dir('/home/jr/