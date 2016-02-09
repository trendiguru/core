# coding: utf-8
import caffe
import os
import sys
import socket
from pylab import *
from caffe import layers as L
from caffe import params as P
from matplotlib import *
sys.path.insert(0,'./python')
sys.path

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

os.chdir('/home/jr/sw/caffe')
with open('examples/mnist/lenet_auto_train.prototxt','w') as f:
    f.write(str(lenet('examples/mnist/mnist_train_lmdb',64)))
with open('examples/mnist/lenet_auto_test.prototxt','w') as f:
    f.write(str(lenet('examples/mnist/mnist_test_lmdb',100)))
host = socket.gethostname()
print('host:'+str(host))
if host == 'jr-ThinkPad-X1-Carbon':
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
imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
print solver.net.blobs['label'].data[:8]
imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
print solver.test_nets[0].blobs['label'].data[:8]
