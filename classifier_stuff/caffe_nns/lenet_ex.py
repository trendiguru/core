# coding: utf-8
import os
print os.getuid() # numeric uid
import pwd
import sys
print pwd.getpwuid(os.getuid()) # full /etc/passwd info


#sys.path.insert(0,'./python')
##sys.path.insert(0,'/usr/lib/python2.7/dist-packages/trendi')
#sys.path.insert(0,'/home/jr/sw/caffe/python')
print(sys.path)

import caffe
import os
import socket
from pylab import *
from caffe import layers as L
from caffe import params as P
#from matplotlib import *
from matplotlib import pyplot as plt

#get_ipython().system(u'data/mnist/get_mnist.sh')
#get_ipython().system(u'examples/mnist/create_mnist.sh')
def lenet(lmdb, batch_size):
    n=caffe.NetSpec()
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

