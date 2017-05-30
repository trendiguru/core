# coding: utf-8
__author__ = 'jeremy'
'''
create nets algorithmically inc. resnet
'''
from pylab import *
import caffe
import caffe.draw
from caffe.proto import caffe_pb2  #...what is this
from caffe import to_proto
from caffe import layers as L
from caffe import params as P
from matplotlib import pyplot as plt
import os
from trendi import Utils
from google.protobuf import text_format

from trendi.classifier_stuff.caffe_nns import lmdb_utils
import math

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

'''
layer {
	bottom: "res5c_branch2b"
	top: "res5c_branch2b"
	name: "bn5c_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: true
	}
}

layer {
	bottom: "res5c_branch2b"
	top: "res5c_branch2b"
	name: "scale5c_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}
'''

def conv(bottom,lr_mult1 = 1,lr_mult2 = 2,decay_mult1=1,decay_mult2 =0,n_output=64,pad='preserve',kernel_size=3,stride=1,weight_filler='xavier',bias_filler='constant',bias_const_val=0.2):
    if pad=='preserve':
        pad = (kernel_size-1)/2
        if float(kernel_size/2) == float(kernel_size)/2:  #kernel size is even
            print('warning: even kernel size, image size cannot be preserved! pad:'+str(pad)+' kernelsize:'+str(kernel_size))
    conv = L.Convolution(bottom,
                        param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                        num_output=n_output,
                        pad = pad,
                        kernel_size=kernel_size,
                        stride = stride,
                        weight_filler=dict(type=weight_filler),
                        bias_filler=dict(type=bias_filler,value=bias_const_val))
    return conv

def conv_relu(bottom,lr_mult1 = 1,lr_mult2 = 2,decay_mult1=1,decay_mult2 =0,n_output=64,pad='preserve',kernel_size=3,stride=1,weight_filler='xavier',bias_filler='constant',bias_const_val=0.2):
    if pad=='preserve':
        pad = (kernel_size-1)/2
        if float(kernel_size/2) == float(kernel_size)/2:  #kernel size is even
            print('warning: even kernel size, image size cannot be preserved! pad:'+str(pad)+' kernelsize:'+str(kernel_size))
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

def conv_bn_scale_relu(bottom, kernel_size=3, num_out=64, stride=1, pad=0, params=[dict(type='msra'),dict(type='constant',value=0)]): #needed for resnet
    '''
    weight_filler = dict(type='msra')
    bias_filler = dict(type='constant', value=0)
    conv_params = [weight_filler, bias_filler]
    :param bottom:
    :param kernel_size:
    :param num_out:
    :param stride:
    :param pad:
    :param params:
    :return:
    '''
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn_train = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                     use_global_stats=False, in_place=True, include=dict(phase=0))
    bn_test = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                     use_global_stats=True, in_place=True, include=dict(phase=1))
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)

    return conv, bn_train, bn_test, scale, relu

def conv_bn_scale(bottom, kernel_size=3, num_out=64, stride=1, pad=0, params=[dict(type='msra'),dict(type='constant',value=0)]): #needed for resnet
    weight_filler = params[0]
    bias_filler = params[1]
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn_train = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                     use_global_stats=False, in_place=True, include=dict(phase=0))
    bn_test = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                     use_global_stats=True, in_place=True, include=dict(phase=1))
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)

    return conv, bn_train, bn_test, scale

def eltsum_relu(bottom1, bottom2): #needed for resnet
    eltsum = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    relu = L.ReLU(eltsum, in_place=True)

    return eltsum, relu

def identity_residual(bottom, kernel_size=3, num_out=64, stride=1, pad=0): #needed for resnet
    conv1, bn1_train, bn1_test, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size=kernel_size, num_out=num_out, stride=stride, pad=pad)
    conv2, bn2_train, bn2_test, scale2 = conv_bn_scale(conv1, kernel_size=kernel_size, num_out=num_out, stride=stride, pad=pad)

    eltsum, relu_after_sum = eltsum_relu(bottom, conv2)

    return conv1, bn1_train, bn1_test, scale1, relu1, conv2, bn2_train, bn2_test, scale2, eltsum, relu_after_sum

def project_residual(bottom, kernel_size=3, num_out=64, stride=1, pad=0): #needed for resnet
    conv_proj, bn_proj_train, bn_proj_test, scale_proj = conv_bn_scale(bottom, kernel_size=3, num_out=num_out, stride=stride, pad=1)

    conv1, bn1_train, bn1_test, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size=kernel_size, num_out=num_out, stride=stride, pad=pad)
    conv2, bn2_train, bn2_test, scale2 = conv_bn_scale(conv1, kernel_size=kernel_size, num_out=num_out, stride=1, pad=pad)

    eltsum, relu_after_sum = eltsum_relu(conv_proj, conv2)

    return conv_proj, bn_proj_train, bn_proj_test, scale_proj, conv1, bn1_train, bn1_test, scale1, relu1, \
           conv2, bn2_train, bn2_test, scale2, eltsum, relu_after_sum

def make_resnet(training_data='cifar10_train', test_data='cifar10_test', mean_file='mean.binaryproto', num_res_in_stage=3):

    num_feature_maps = np.array([16, 32, 64]) # feature map size: [32, 16, 8]

    n = caffe.NetSpec()
    n.data, n.label = L.Data(source=training_data, backend=P.Data.LEVELDB, batch_size=128, ntop=2,
                                     transform_param=dict(crop_size=32, mean_file=mean_file, mirror=True),
                                     image_data_param=dict(shuffle=True), include=dict(phase=0))
    n.test_data, n.test_label = L.Data(source=test_data, backend=P.Data.LEVELDB, batch_size=100, ntop=2,
                                     transform_param=dict(crop_size=32, mean_file=mean_file, mirror=False),
                                     include=dict(phase=1))
    n.conv1, n.bn_conv1_train, n.bn_conv1_test, n.scale_conv1, n.relu_conv1 = \
                      conv_bn_scale_relu(n.data, kernel_size=3, num_out=16, stride=1, pad=1, params=conv_params)

    last_stage = 'n.relu_conv1'

    for num_map in num_feature_maps:
        num_map = int(num_map)
        for res in list(range(num_res_in_stage)):
            stage = 'map' + str(num_map) + '_' + str(res + 1) + '_'
            if np.where(num_feature_maps == num_map)[0] >= 1 and res == 0:
                make_res = 'n.' + stage + 'conv_proj,' + \
                           'n.' + stage + 'bn_proj_train,' + \
                           'n.' + stage + 'bn_proj_test,' + \
                           'n.' + stage + 'scale_proj,' + \
                           'n.' + stage + 'conv_a,' + \
                           'n.' + stage + 'bn_a_train, ' + \
                           'n.' + stage + 'bn_a_test, ' + \
                           'n.' + stage + 'scale_a, ' + \
                           'n.' + stage + 'relu_a, ' + \
                           'n.' + stage + 'conv_b, ' + \
                           'n.' + stage + 'bn_b_train, ' + \
                           'n.' + stage + 'bn_b_test, ' + \
                           'n.' + stage + 'scale_b, ' + \
                           'n.' + stage + 'eltsum, ' + \
                           'n.' + stage + 'relu_after_sum' + \
                           ' = project_residual(' + last_stage + ', num_out=num_map, stride=2, pad=1)'
                exec(make_res)
                last_stage = 'n.' + stage + 'relu_after_sum'
                continue

            make_res = 'n.' + stage + 'conv_a, ' + \
                       'n.' + stage + 'bn_a_train, ' + \
                       'n.' + stage + 'bn_a_test, ' + \
                       'n.' + stage + 'scale_a, ' + \
                       'n.' + stage + 'relu_a, ' + \
                       'n.' + stage + 'conv_b, ' + \
                       'n.' + stage + 'bn_b_train, ' + \
                       'n.' + stage + 'bn_b_test, ' + \
                       'n.' + stage + 'scale_b, ' + \
                       'n.' + stage + 'eltsum, ' + \
                       'n.' + stage + 'relu_after_sum' + \
                       ' = identity_residual(' + last_stage + ', num_out=num_map, stride=1, pad=1)'
            exec(make_res)
            last_stage = 'n.' + stage + 'relu_after_sum'

    exec('n.pool_global = L.Pooling(' + last_stage + ', pool=P.Pooling.AVE, global_pooling=True)')
    n.score = L.InnerProduct(n.pool_global, num_output=10,
                                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                          weight_filler=dict(type='gaussian', std=0.01),
                                          bias_filler=dict(type='constant', value=0))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()

def build_resnet(N=9,training_data_path=None,test_data_path=None,mean_file_path=None,save_file='resnet_pygen'):
    '''
    #from https://github.com/Coldmooon/ResNet-Prototxt-for-Caffe/blob/master/resnet_cifar_generator.py
    This script is used to creat ResNet prototxt for Caffe.
    Following the original paper, N = {3, 5, 7 ,9} needs to be given, where
    3  for 20-layer network
    5  for 32-layer network
    7  for 44-layer network
    9  for 56-layer network
    18 for 110-layer network
    Usage: <option(s)> N
    python resnet_generator.py training_data_path test_data_path mean_file_path N
    Options:
    training_data_path: the path of training data (LEVELDB or LMDB).
       test_data_path: the path of test data (LEVELDB or LMDB).
       mean_file_path: the path of mean file for training data.
                    N: a parameter introduced by the original paper, meaning the number of repeat of residual
                       building block for each feature map size (32, 16, 8).
                       For example, N = 5 means that creat 5 residual building blocks for feature map size 32,
                       5 for feature map size 16, and 5 for feature map size 8. Besides, in each building block,
                       two weighted layers are included. So there are (5 + 5 + 5)*2 + 2 = 32 layers.
    :param N:
    :return:
    '''

    # training_data_path = str(sys.argv[1])
    # test_data_path = str(sys.argv[2])
    # mean_file_path = str(sys.argv[3])
    # N =  int(sys.argv[4])

    proto_created = str(make_resnet(training_data_path, test_data_path, mean_file_path, N))
    proto_created = proto_created.replace('_test"', '"')
    proto_created = proto_created.replace('_train"', '"')
    restnet_prototxt = proto_created.replace('test_', '')

    save_file = save_file + str(6 * N + 2) + '.prototxt'
    with open(save_file, 'w') as f:
        f.write(restnet_prototxt)

    print('Saved ' + save_file)

# another helper function
def fc_relu(bottom, nout,lr_mult1=1,decay_mult1=1,lr_mult2=2,decay_mult2=0):
    fc = L.InnerProduct(bottom,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],num_output=nout,weight_filler=dict(type='xavier'))
    relu = L.ReLU(fc,in_place=True)
#    return fc, L.ReLU(fc, in_place=True)
    return fc,relu

def batchnorm(bottom,stage='train',in_place=True):
    batch_norm = L.BatchNorm(bottom, in_place=in_place, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                              batch_norm_param={'use_global_stats': stage=='test'})
    scale = L.Scale(batch_norm, bias_term=True, in_place=in_place)
    return batch_norm,scale

def conv_relu_bn(bottom, n_output, lr_mult1 = 1,lr_mult2 = 2,decay_mult1=1,decay_mult2 =0, kernel_size=1, stride=1, pad='preserve',stage='train',bias_filler_type='constant',bias_const_val=0.2):
    if pad=='preserve':
        pad = (kernel_size-1)/2
        if float(kernel_size/2) == float(kernel_size)/2:  #kernel size is even
            print('warning: even kernel size, image size cannot be preserved! pad:'+str(pad)+' kernelsize:'+str(kernel_size))
    conv = L.Convolution(bottom,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                kernel_size=kernel_size, stride=stride,
                num_output=n_output, pad=pad, bias_filler=dict(type=bias_filler_type,value=bias_const_val), weight_filler=dict(type='xavier'))
    # see https://groups.google.com/forum/#!topic/caffe-users/h4E6FV_XkfA - verify this if poss
    relu = L.ReLU(conv, in_place=True)
    #batch norm after relu better according to this https://github.com/ducha-aiki/caffenet-benchmark/issues/3
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                             batch_norm_param={'use_global_stats': stage=='test'})
    scale = L.Scale(conv, bias_term=True, in_place=True)
    return conv,relu,batch_norm,scale

def Inception7A(data, num_1x1, num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5, pool, proj):
    tower_1x1 = Conv(data, 1, num_1x1)
    tower_5x5 = Conv(data, 1, num_5x5_red)
    tower_5x5 = Conv(tower_5x5, 5, num_5x5, 1, 2)
    tower_3x3 = Conv(data, 1, num_3x3_red)
    tower_3x3 = Conv(tower_3x3, 3, num_3x3_1, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_2')
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
    concat = mx.sym.Concat(*[tower_1x1, tower_5x5, tower_3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def residual_factory1(bottom, num_filter):
    conv1 = conv_factory_relu(bottom, 3, num_filter, 1, 1);
    conv2 = conv_factory(conv1, 3, num_filter, 1, 1);
    residual = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def residual_factory_proj(bottom, num_filter, stride=2):
    conv1 = conv_factory_relu(bottom, 3, num_filter, stride, 1);
    conv2 = conv_factory(conv1, 3, num_filter, 1, 1);
    proj = conv_factory(bottom, 1, num_filter, stride, 0);
    residual = L.Eltwise(conv2, proj, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def resnet(train_lmdb, test_lmdb, batch_size=256, stages=[2, 2, 2, 2], first_output=32, include_acc=False):
    # now, this code can't recognize include phase, so there will only be a TEST phase data layer
    data, label = L.Data(source=train_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TEST')))

    # the net itself
    relu1 = conv_factory_relu(data, 3, first_output, stride=1, pad=1)
    relu2 = conv_factory_relu(relu1, 3, first_output, stride=1, pad=1)
    residual = max_pool(relu2, 3, stride=2)

    for i in stages[1:]:
        first_output *= 2
        for j in range(i):
            if j==0:
                if i==0:
                    residual = residual_factory_proj(residual, first_output, 1)
                else:
                    residual = residual_factory_proj(residual, first_output, 2)
            else:
                residual = residual_factory1(residual, first_output)

    glb_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True);
    fc = L.InnerProduct(glb_pool, num_output=1000)
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)
#
def jr_resnet_test(n_bs = [2,3,5,2],source='trainfile',batch_size=10,nout_initial=64,
                 lr_mult=(1,1),weight_filler='xavier',use_global_stats=False): #global stats false for train, true for test/deploy

    '''
    resnet 50: n_bs = [2,3,5,2]  this
    :param n_bs: number of 'B' units for each 'A' unit
    :param lr_mult:
    :param decay_mult:
    :param weight_filler:
    :return:
    '''
    data, label = L.Data(source=source, batch_size=batch_size, ntop=2)
    transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True)
    # the net itself
    stride=2
    kernel_size = 7
    pad = 3
    conv = L.Convolution(data, kernel_size=kernel_size, stride=stride,
                                num_output=nout_initial, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
#    n_neurons = (W-F+2P)/S + 1  W-orig width, F-filter size(kernel), P-pad S-stride
    batch_norm = L.BatchNorm(conv, in_place=True)
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)

    loss = L.SoftmaxWithLoss(relu, label)
    acc = L.Accuracy(relu, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    print(type(loss),type(acc))
    return to_proto(acc)


def jr_resnet_u(n_bs=[2,3,5,2],source='trainfile',batch_size=10,nout_initial=64,
                 lr_mult=(1,1),decay_mult=(2,0),weight_filler='xavier',use_global_stats=False,image_dims=(256,256)):
    #going with 256 as a power of 2 making upsamling cleaner (maxpool on 27 leads to 14 and upsampling 14 leads to 28...)
    #global stats false for train, true for test/deploy, possibly can be left out and default is ok
    '''
    resnet 50: n_bs = [2,3,5,2]
    a Unet based on resnet with crossconnections from  data and every final B (i think)
    :param n_bs: number of 'B' units for each 'A' unit
    :param lr_mult:
    :param decay_mult:
    :param weight_filler:
    :return:
    '''
    #cross-connection layers
    l_cross = [None for k in range(len(n_bs)+2)]
    current_cross_layer = 0
    current_dims = np.array(image_dims)
    data, label = L.Data(source=source, batch_size=batch_size, ntop=2) #see if tihs can be changed to python datalayer
    transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True)
    l_cross[current_cross_layer] = data
    current_cross_layer+=1
    # the net itself
    stride=2
    kernel_size = 7
    pad = 3
    conv = L.Convolution(data, kernel_size=kernel_size, stride=stride,
                                num_output=nout_initial, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
#    n_neurons = (W-F+2P)/S + 1  W-orig width, F-filter size(kernel), P-pad S-stride
    current_dims = (current_dims-kernel_size+2*pad)/stride + 1 # W-orig width, F-filter size(kernel), P-pad S-stride
    print('dims after conv1 '+str(current_dims)+' originally '+str(image_dims))
    batch_norm = L.BatchNorm(conv, in_place=True)
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)

    l_cross[current_cross_layer] = conv  #could do this before the bn/relu
    current_cross_layer+=1

    kernel_size=3
    residual = max_pool(relu, kernel_size, stride=2)

  #  relu1 = conv_factory_relu(data, nout_initial, kernel_sizes = (1,7), stride=1)
 #   relu2 = conv_factory_relu(relu1, nout_initial, kernel_size=3, stride=1)
    kernel_size=3
    pad = 1
#    n_neurons = (W-F+2P)/S + 1  W-orig width, F-filter size(kernel), P-pad S-stride
    current_dims = np.divide(current_dims-kernel_size+2*pad,stride)+1
    print('dims after maxpool1 '+str(current_dims))

    # starting the U - going in

    ##########initial AB...B  (stride (1,1) )
    nout = 64
    kernel_sizes = (1,3)
    strides = (1,1)
    l = jr_resnet_A(residual,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
    print('doing {} Bs for initial A, nout {}'.format(n_bs[0],nout))
    for j in range(n_bs[0]-1):
        l = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
    l_cross[current_cross_layer] = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
    l = l_cross[current_cross_layer]
    current_cross_layer += 1


    ##########remaining AB...B's (stride(2,1) for A and (1,1) for B)
    for i in range(1,len(n_bs)):
        nout = nout * 2
        print('doing {} Bs for A[{}], nout {}'.format(n_bs[i],i,nout))
        strides = (2,1)
        l = jr_resnet_A(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
        #the A sequence  strides by both
        stride = strides[0]*strides[1] #aka 2
        current_dims = np.divide(current_dims-kernel_size+2*pad,stride)+1
        print('dims after A{}:{}'.format(i,current_dims))
        strides = (1,1)
        for j in range(n_bs[i]-1):
            l = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
        l_cross[current_cross_layer] = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
        l = l_cross[current_cross_layer]
        current_cross_layer += 1
        final_cross_index = i

    current_cross_layer-=1
    pad = 3
    kernel_size = 8
    stride = 1
    print('dims before avgpool '+str(current_dims))
    residual = L.Pooling(l, pool=P.Pooling.AVE, kernel_size=kernel_size, stride=stride)
#    n_neurons = (W-F+2P)/S + 1  W-orig width, F-filter size(kernel), P-pad S-stride

    #bottom of U
    current_dims = np.divide(current_dims-kernel_size+2*pad,stride)+1
    print('dims after maxpool2 '+str(current_dims))

    n_output_filters = math.ceil(float(nout)/(current_dims[0]*current_dims[1])) #arbitrary
    n_neurons = int(math.ceil(current_dims[0]*current_dims[1]*n_output_filters)) *2
    print('orig filters {} x {} y {} n_filt {} neurons {} '.format(nout,current_dims[0],current_dims[1],n_output_filters,n_neurons))
    fc = L.InnerProduct(residual,param= \
                        [dict(lr_mult=lr_mult[0]),
                         dict(lr_mult=lr_mult[1])],
                        weight_filler=dict(type=weight_filler),
                        num_output=n_neurons)
    relu = L.ReLU(fc, in_place=True)

    # loss = L.SoftmaxWithLoss(relu, label)
    # acc = L.Accuracy(relu, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    # return to_proto(loss, acc)


    reshape = L.Reshape(relu, reshape_param = dict(shape=dict(dim=[0,-1,current_dims[0],current_dims[1]])))     # batchsize X infer X 7 X 7 , infer should=6272/49=128
    l = reshape
    raw_input('ret to cont')

    #Rest of U - going back up
    kernel_sizes = (1,3)
    strides = (1,1)
    for i in range(len(n_bs)-1,0,-1):
        #get the cross
        print('doing cross for {} with layers {} and {}'.format(current_cross_layer,l_cross[current_cross_layer],reshape))
        bottom = [l_cross[current_cross_layer], l]
        current_cross_layer -= 1
        l = L.Concat(*bottom) #param=dict(concat_dim=1))
        strides = (1,1) #keep stride at 1 to avoid downsample like on way in
        l = jr_resnet_A(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
        strides = (1,1)
        print('doing {} Bs for A[{}], nout {}'.format(n_bs[i],i,nout))
        for j in range(n_bs[i]):
            l = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
        nout=nout/2
        kernel_size = 2
        stride = 2
        pad = 0
        initial_deconv_value = 1.0/(stride*stride)
        print('initial deconv value '+str(initial_deconv_value))
        deconv = L.Deconvolution(l,
                                param=[dict(lr_mult=lr_mult[0],decay_mult=decay_mult[0]),dict(lr_mult=lr_mult[1],decay_mult=decay_mult[1])],
    #                            num_output=64,
                                convolution_param = dict(num_output=nout, pad = 0,
                                kernel_size=kernel_size,
                                stride = stride,
    #                            weight_filler= {'type':'xavier'},
                                weight_filler= {'type':'constant','value':initial_deconv_value},
                                bias_filler= {'type':'constant','value':0.0}) )
        l=deconv
        current_dims = stride*(current_dims-1) + kernel_size - 2 * pad
        print('dims after deconv1 '+str(current_dims))




#       stride_data[i] * (input_dim - 1)  + kernel_extent - 2 * pad_data[i];
#    current_dims = np.divide(current_dims-kernel_size+2*pad,stride)+1  #
    # see https://github.com/BVLC/caffe/blob/master/src/caffe/layers/deconv_layer.cpp#L10  for deconv
    current_dims = stride*(current_dims-1) + kernel_size - 2 * pad
    print('dims after deconv2 '+str(current_dims))

#    n_neurons = (W-F+2P)/S + 1  W-orig width, F-filter size(kernel), P-pad S-stride

    # fc = L.InnerProduct(deconv2,param= \
    #                     [dict(lr_mult=lr_mult[0]),
    #                      dict(lr_mult=lr_mult[1])],
    #                     weight_filler=dict(type=weight_filler),
    #                     num_output=2)

    kernel_size = 2
    stride = 2
    pad = 0

#use this eg in a subsequent conv or two
    deconv = L.Deconvolution(l,
                            param=[dict(lr_mult=lr_mult[0],decay_mult=decay_mult[0]),dict(lr_mult=lr_mult[1],decay_mult=decay_mult[1])],
#                            num_output=64,
                            convolution_param = dict(num_output=nout, pad = 0,
                            kernel_size=kernel_size,
                            stride = stride,
#                            weight_filler= {'type':'xavier'},
                            weight_filler= {'type':'constant','value':initial_deconv_value},
                            bias_filler= {'type':'constant','value':0.0}) )
    l=deconv


    loss = L.SoftmaxWithLoss(l, label)
    acc = L.Accuracy(l, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    #
    #
    # loss = L.SoftmaxWithLoss(l, label)
    # acc = L.Accuracy(l, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    print(type(loss),type(acc))
    return to_proto(loss, acc)

def jr_resnet(n_bs = [2,3,5,2],source='trainfile',batch_size=10,nout_initial=64,
                 lr_mult=(1,1),weight_filler='xavier',use_global_stats=False): #global stats false for train, true for test/deploy
    '''

    resnet 50: n_bs = [2,3,5,2]  this
    :param n_bs: number of 'B' units for each 'A' unit
    resnet is composed of what you could describe in regex as (AB+)+
    e.g. ABBABBBABBBBBABB , A and B units built in own functions below
    see http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
    :param source:
    :param batch_size:
    :param nout_initial:
    :param lr_mult:
    :param weight_filler:
    :param use_global_stats:
    :return:
    '''
    data, label = L.Data(source=source, batch_size=batch_size, ntop=2)
#    transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True)

    #not sure why this was originally 227 but that number pops up every once in a while, netscope says 224
    #and its a multiple of 32 which gpus like so 224 it is
    transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True)
    # the net itself
    conv = L.Convolution(data, kernel_size=7, stride=2,
                                num_output=nout_initial, pad=3, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True,use_global_stats=use_global_stats)
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)

    # loss = L.SoftmaxWithLoss(relu, label)
    # acc = L.Accuracy(relu, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    # print(type(loss),type(acc))
    # return to_proto(acc)


  #  relu1 = conv_factory_relu(data, nout_initial, kernel_sizes = (1,7), stride=1)
 #   relu2 = conv_factory_relu(relu1, nout_initial, kernel_size=3, stride=1)
    residual = max_pool(relu, 3, stride=2)
    # n.b. this is being done without a pad

    #First AB+ unit - different only since its bottom is different, could make into one by an if
    nout = 64
    kernel_sizes = (1,3)
    strides = (1,1)
    l = jr_resnet_A(residual,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
    print('doing {} Bs for initial A'.format(n_bs[0]))
    for j in range(n_bs[0]):
        l = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)

    #all subsequent AB+ units
    for i in range(1,len(n_bs)):
        nout = nout * 2
        print('doing {} Bs for A[{}], nout {}'.format(n_bs[i],i,nout))
        strides = (2,1)
        l = jr_resnet_A(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)
        strides = (1,1)
        for j in range(n_bs[i]):
            l = jr_resnet_B(l,nout=nout,kernel_sizes=kernel_sizes,strides=strides,use_global_stats=use_global_stats)

    #    residual = max_pool(l, 7, stride=1)
    residual = L.Pooling(l, pool=P.Pooling.AVE, kernel_size=7, stride=1)

    fc = L.InnerProduct(residual,param= \
                        [dict(lr_mult=lr_mult[0]),
                         dict(lr_mult=lr_mult[1])],
                        weight_filler=dict(type=weight_filler),
                        num_output=1000)

    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)

def jr_resnet_A_deconv(bottom,nout,kernel_sizes=(1,3),strides=(1,1),use_global_stats=False):
    #kernel_sizes[1] is the middle (larger) kernel size
    #strides[0] is the first (sometimes larger) stride
    if strides[0] == 1 and strides[1] == 1: #if all strides are 1 then no need for deconv
        return jr_resnet_A(bottom,nout,kernel_sizes=(1,3),strides=(1,1),use_global_stats=False)
    cbsr_b2_a = conv_factory_relu(bottom, nout, kernel_size=kernel_sizes[0],stride=strides[0],use_global_stats=use_global_stats) #CBSR
    cbsr_b2_b = conv_factory_relu(cbsr_b2_a, nout, kernel_size=kernel_sizes[1],stride=strides[1],use_global_stats=use_global_stats)
    n_cbs = nout * 4
    cbs_b2_c = conv_factory(cbsr_b2_b, n_cbs,kernel_size=kernel_sizes[0],stride=strides[1],use_global_stats=use_global_stats) #CBS
    cbs_b1_a = conv_factory(bottom, n_cbs, kernel_size=kernel_sizes[0],stride=strides[0],use_global_stats=use_global_stats)
    residual = L.Eltwise(cbs_b1_a, cbs_b2_c, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def jr_resnet_A(bottom,nout,kernel_sizes=(1,3),strides=(1,1),use_global_stats=False):
    #kernel_sizes[1] is the middle (larger) kernel size
    #strides[0] is the first (sometimes larger) stride
    cbsr_b2_a = conv_factory_relu(bottom, nout, kernel_size=kernel_sizes[0],stride=strides[0],use_global_stats=use_global_stats) #CBSR
    cbsr_b2_b = conv_factory_relu(cbsr_b2_a, nout, kernel_size=kernel_sizes[1],stride=strides[1],use_global_stats=use_global_stats)
    n_cbs = nout * 4
    cbs_b2_c = conv_factory(cbsr_b2_b, n_cbs,kernel_size=kernel_sizes[0],stride=strides[1],use_global_stats=use_global_stats) #CBS
    cbs_b1_a = conv_factory(bottom, n_cbs, kernel_size=kernel_sizes[0],stride=strides[0],use_global_stats=use_global_stats)
    residual = L.Eltwise(cbs_b1_a, cbs_b2_c, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def jr_resnet_B(bottom,nout,kernel_sizes=(1,3),strides=(1,1),use_global_stats=False):
    #kernel_sizes[1] is the middle (larger) kernel size
    #strides[0] is the first (sometimes larger) stride
    cbsr_b2_a = conv_factory_relu(bottom, nout, kernel_size=kernel_sizes[0],stride=strides[0],use_global_stats=use_global_stats) #CBSR
    cbsr_b2_b = conv_factory_relu(cbsr_b2_a, nout, kernel_size=kernel_sizes[1],stride=strides[1],use_global_stats=use_global_stats)
    n_cbs = nout * 4
    cbs_b2_c = conv_factory(cbsr_b2_b, n_cbs,kernel_size=kernel_sizes[0],stride=strides[1],use_global_stats=use_global_stats) #CBS
    residual = L.Eltwise(bottom,cbs_b2_c, operation=P.Eltwise.SUM)
    relu = L.ReLU(residual, in_place=True)
    return relu

def conv_factory(bottom, nout,kernel_size=1, stride=1, pad='preserve',filler='msra',use_global_stats=False): #CBS
    if pad=='preserve':
        pad = (kernel_size-1)/2
        if float(kernel_size/2) == float(kernel_size)/2:  #kernel size is even
            print('warning: even kernel size, image size cannot be preserved! pad:'+str(pad)+' kernelsize:'+str(kernel_size))
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type=filler))
#    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],batch_norm_param=dict(use_global_stats=use_global_stats))
    batch_norm = L.BatchNorm(conv, in_place=True)#apparently, default global_param and lr is ok
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

def conv_factory_relu(bottom, nout, kernel_size=1, stride=1, pad='preserve',filler='msra',use_global_stats=False): #CBSR
    if pad=='preserve':
        pad = (kernel_size-1)/2
        if float(kernel_size/2) == float(kernel_size)/2:  #kernel size is even
            print('warning: even kernel size, image size cannot be preserved! pad:'+str(pad)+' kernelsize:'+str(kernel_size))
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type=filler))
#    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],batch_norm_param=dict(use_global_stats=use_global_stats))
    batch_norm = L.BatchNorm(conv, in_place=True) #default global_param and lr is supposed to be ok
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def residual_factory1(bottom, num_filter):
    conv1 = conv_factory_relu_inverse_no_inplace(bottom, 3, num_filter, 1, 1);
    conv2 = conv_factory_relu_inverse(conv1, 3, num_filter, 1, 1);
    addition = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    return addition

def residual_factory_proj(bottom, num_filter, stride=2):
    batch_norm = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    conv1 = conv_factory_relu(scale, 3, num_filter, stride, 1);
    conv2 = L.Convolution(conv1, kernel_size=3, stride=1,
                                num_output=num_filter, pad=1, weight_filler=dict(type='msra'));
    proj = L.Convolution(scale, kernel_size=1, stride=stride,
                                num_output=num_filter, pad=0, weight_filler=dict(type='msra'));
    addition = L.Eltwise(conv2, proj, operation=P.Eltwise.SUM)
    return addition

def conv_factory_relu_inverse(bottom, ks, nout, stride=1, pad=0):
    batch_norm = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    return conv

def conv_factory_relu_inverse_no_inplace(bottom, ks, nout, stride=1, pad=0):
    batch_norm = L.BatchNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    return conv

def max_pool(bottom, ks, stride=1):
    '''note this can take a pad '''
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


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

#    n.conv1_1 = conv_factory_relu(n.data,ks=3,nout=64,stride=1,pad=1)
    n.conv1_1,n.relu1_1 = conv_relu(n.data,n_output=64,kernel_size=3,pad=1)
    n.conv1_2,n.relu1_2 = conv_relu_bn(n.conv1_1,n_output=64,kernel_size=3,pad=1)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2_1,n.relu2_1 = conv_relu(n.pool1,n_output=128,kernel_size=3,pad=1)
    n.conv2_2,n.relu2_2 = conv_relu_bn(n.conv2_1,n_output=128,kernel_size=3,pad=1)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv3_1,n.relu3_1 = conv_relu(n.pool2,n_output=256,kernel_size=3,pad=1)
    n.conv3_2,n.relu3_2 = conv_relu(n.conv3_1,n_output=256,kernel_size=3,pad=1)
    n.conv3_3,n.relu3_3 = conv_relu_bn(n.conv3_2,n_output=256,kernel_size=3,pad=1)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv4_1,n.relu4_1 = conv_relu(n.pool3,n_output=512,kernel_size=3,pad=1)
    n.conv4_2,n.relu4_2 = conv_relu(n.conv4_1,n_output=512,kernel_size=3,pad=1)
    n.conv4_3,n.relu4_3 = conv_relu_bn(n.conv4_2,n_output=512,kernel_size=3,pad=1)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv5_1,n.relu5_1 = conv_relu(n.pool4,n_output=512,kernel_size=3,pad=1)
    n.conv5_2,n.relu5_2 = conv_relu(n.conv5_1,n_output=512,kernel_size=3,pad=1)
    n.conv5_3,n.relu5_3 = conv_relu_bn(n.conv5_2,n_output=512,kernel_size=3,pad=1)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.fc6 = L.InnerProduct(n.pool5,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.relu6 = L.ReLU(n.fc6, in_place=True)
    n.drop6 = L.Dropout(n.fc6, dropout_param=dict(dropout_ratio=0.5),in_place=True)

    n.fc7 = L.InnerProduct(n.fc6,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
    n.relu7 = L.ReLU(n.fc7, in_place=True)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.5),in_place=True)

    n.fc8 = L.InnerProduct(n.fc7,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=1000)
    return n.to_proto()

def test_convbnrelu(db,mean_value=[112.0,112.0,112.0],imsize=(224,224),n_cats=21):
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
    n.conv,n.relu,n.scale = conv_relu_bn(n.data,n_output=64,kernel_size=3,pad='preserve')
    return n.to_proto()

def sharpmask(db,mean_value=[112.0,112.0,112.0],imsize=(224,224),n_cats=21,stage='train'):
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

    n.bn1,n.scale1 = batchnorm(n.data,stage=stage)
    n.conv1_1,n.relu1_1 = conv_relu_bn(n.scale1,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_2,n.relu1_2 = conv_relu_bn(n.conv1_1,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 112x112
    n.conv2_1,n.relu2_1 = conv_relu_bn(n.pool1,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_2,n.relu2_2 = conv_relu_bn(n.conv2_1,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 56x56
    n.conv3_1,n.relu3_1 = conv_relu_bn(n.pool2,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_2,n.relu3_2 = conv_relu_bn(n.conv3_1,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_3,n.relu3_3 = conv_relu_bn(n.conv3_2,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 28x28
    n.conv4_1,n.relu4_1 = conv_relu_bn(n.pool3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_2,n.relu4_2 = conv_relu_bn(n.conv4_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_3,n.relu4_3 = conv_relu_bn(n.conv4_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 14x14
    n.conv5_1,n.relu5_1 = conv_relu_bn(n.pool4,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_2,n.relu5_2 = conv_relu_bn(n.conv5_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_3,n.relu5_3 = conv_relu_bn(n.conv5_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 7x7
    #convolutional - kernelsize of HxW will not suffice, 2Hx2W actuallyrequired to simulate fc
 #   n.conv6_1,n.relu6_1 = conv_relu(n.pool5,n_output=4096,kernel_size=15,pad=7)
       #instead of L.InnerProduct(n.pool5,param=[dict(lr_mult=lr_mult1),dict(lr_mult=lr_mult2)],weight_filler=dict(type='xavier'),num_output=4096)
#    n.drop6_1 = L.Dropout(n.conv6_1, dropout_param=dict(dropout_ratio=0.5),in_place=True)
    #try nonconvolutional.

    n.fc6,n.relu6 = fc_relu(n.pool5,3136)  #6272=7*7*128
    n.bn6,n.scale6 = batchnorm(n.fc6,stage=stage)
    n.drop6 = L.Dropout(n.fc6, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    n.fc7,n.relu7 = fc_relu(n.fc6,3136)
    n.bn7,n.scale7 = batchnorm(n.fc7,stage=stage)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.2),in_place=True)

#layer {
#    name: "reshape"
#    type: "Reshape"
#    bottom: "fc3"
#    top: "reshape"
#    reshape_param {
#      shape {
#        dim: 0  # copy the dimension from below
#        dim: 8
#        dim: 64
#        dim: 64
#      }
#    }
#}
    #the following will be 7x7 (original /32).
#    n.reshape8 = L.Reshape(n.fc7,
#                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                            reshape_param = {'shape':{'dim':[0,128,7,7] }})    # batchsize X infer X 7 X 7 , infer should=6272/49=128
    n.reshape8 = L.Reshape(n.fc7, reshape_param = dict(shape=dict(dim=[0,-1,7,7])))     # batchsize X infer X 7 X 7 , infer should=6272/49=128

#    n.resh = L.Reshape(n.fc3, reshape_param={'shape':{'dim': [1, 1, 64, 64]}})

    #from https://github.com/BVLC/caffe/issues/4052
    #n.deconv = L.Deconvolution(n.input,
    #convolution_param=dict(num_output=21, kernel_size=64, stride=32))
    n.conv8_0,n.relu8_0 = conv_relu_bn(n.reshape8,n_output=512,kernel_size=7,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...


    #the following will be 14x14 (original /16).
    n.deconv8 = L.Deconvolution(n.conv8_0,
                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                            num_output=64,
                            convolution_param = dict(num_output=512, pad = 0,
                            kernel_size=2,
                            stride = 2,
                            weight_filler= {'type':'xavier'},
                            bias_filler= {'type':'constant','value':0.2}) )

#    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
#                                num_output=n_output, pad=pad, bias_term=False, weight_filler=dict(type='msra'))

    n.conv8_1,n.relu8_1 = conv_relu_bn(n.deconv8,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv8_cross1,n.relu8_cross1 = conv_relu_bn(n.conv5_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv8_cross2,n.relu8_cross2 = conv_relu_bn(n.conv8_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv8_cross2, n.conv8_1]
    n.cat8 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv8_2,n.relu8_2 = conv_relu(n.cat8,n_output=512,kernel_size=5,pad='preserve')


#yes 9 is missing, sue me

#the following will be 28x28  (original /16)
#deconv doesnt work from python , so these need to be changed by hand #
    # this is the 'more efficient equivalent' as listed in fb paper, except with extra relu's . try strict rewrite if this doesnt work

    n.deconv10 = L.Deconvolution(n.conv8_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=512,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))


    n.conv10_1,n.relu10_1 = conv_relu_bn(n.deconv10,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
#    n.conv10_2 = conv_bn_relu(n.conv10_1,n_output=512,kernel_size=3,pad='preserve')  #watch out for padsize here, make sure outsize is 14x14 #indeed
    n.conv4_cross1,n.relu4_cross1 = conv_relu_bn(n.conv4_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_cross2,n.relu4_cross2 = conv_relu_bn(n.conv4_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv4_cross2, n.conv10_1]
    n.cat10 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv10_2,n.relu10_2 = conv_relu(n.cat10,n_output=512,kernel_size=3,pad='preserve')

    #the following will be 56x56  (original /4)
    n.deconv11 = L.Deconvolution(n.conv10_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=256,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv11_1,n.relu11_1 = conv_relu_bn(n.deconv11,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.conv3_cross1,n.relu3_cross1 = conv_relu_bn(n.conv3_3,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_cross2,n.relu3_cross2 = conv_relu_bn(n.conv3_cross1,n_output=256,kernel_size=3,pad='preserve')
    bottom=[n.conv3_cross2, n.conv11_1]
    n.cat11 = L.Concat(*bottom)
    n.conv11_2,n.relu11_2 = conv_relu_bn(n.cat11,n_output=256,kernel_size=3,pad='preserve')

    #the following will be 112x112  (original /4)
    n.deconv12 = L.Deconvolution(n.conv11_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=128,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv12_1,n.relu12_1 = conv_relu_bn(n.deconv12,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.conv2_cross1,n.relu2_cross1 = conv_relu_bn(n.conv2_2,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_cross2,n.relu2_cross2 = conv_relu_bn(n.conv2_cross1,n_output=128,kernel_size=3,pad='preserve')
    bottom=[n.conv2_cross2, n.conv12_1]
    n.cat12 = L.Concat(*bottom)
    n.conv12_2,n.relu12_2 = conv_relu_bn(n.cat12,n_output=128,kernel_size=3,pad='preserve')


    #the following will be 224x224  (original /2)
    n.deconv13 = L.Deconvolution(n.conv12_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=64,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv13_1,n.relu13_1 = conv_relu_bn(n.deconv13,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.conv1_cross1,n.relu1_cross1 = conv_relu_bn(n.conv1_2,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_cross2,n.relu1_cross2 = conv_relu_bn(n.conv1_cross1,n_output=64,kernel_size=3,pad='preserve')
    bottom=[n.conv1_cross2, n.conv13_1]


    n.cat13 = L.Concat(*bottom)
    n.conv13_2,n.relu13_2 = conv_relu_bn(n.cat13,n_output=64,kernel_size=3,pad='preserve',stage=stage)  #this is halving N_filters

    n.conv_final = conv(n.conv13_2,n_output=n_cats,kernel_size=3,pad='preserve')

#    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label,normalize=True)
    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label)

#    n.deconv1 = L.Deconvolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                convolution_param=[dict(num_output=512,bias_term=False,kernel_size=2,stride=2)])
    return n.to_proto()

def sharp2(db,mean_value=[112.0,112.0,112.0],imsize=(224,224),n_cats=21,stage='train'):
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

    n.bn1,n.scale1 = batchnorm(n.data,stage=stage)
    n.conv1_1,n.relu1_1,n.bn1_1,n.scale1_1 = conv_relu_bn(n.scale1,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_2,n.relu1_2,n.bn1_2,n.scale1_2 = conv_relu_bn(n.conv1_1,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 112x112
    n.conv2_1,n.relu2_1,n.bn2_1,n.scale2_1 = conv_relu_bn(n.pool1,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_2,n.relu2_2,n.bn2_2,n.scale2_2 = conv_relu_bn(n.conv2_1,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 56x56
    n.conv3_1,n.relu3_1,n.bn3_1,n.scale3_1 = conv_relu_bn(n.pool2,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_2,n.relu3_2,n.bn3_2,n.scale3_2 = conv_relu_bn(n.conv3_1,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_3,n.relu3_3,n.bn3_3,n.scale3_3 = conv_relu_bn(n.conv3_2,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 28x28
    n.conv4_1,n.relu4_1,n.bn4_1,n.scale4_1 = conv_relu_bn(n.pool3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_2,n.relu4_2,n.bn4_2,n.scale4_2 = conv_relu_bn(n.conv4_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_3,n.relu4_3,n.bn4_3,n.scale4_3 = conv_relu_bn(n.conv4_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    #the following will be 14x14
    n.conv5_1,n.relu5_1,n.bn5_1,n.scale5_1 = conv_relu_bn(n.pool4,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_2,n.relu5_2,n.bn5_2,n.scale5_2 = conv_relu_bn(n.conv5_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_3,n.relu5_3,n.bn5_3,n.scale5_3 = conv_relu_bn(n.conv5_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.fc6,n.relu6 = fc_relu(n.pool5,3136)  #6272=7*7*128
    n.bn6,n.scale6 = batchnorm(n.fc6,stage=stage)
    n.drop6 = L.Dropout(n.fc6, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    n.fc7,n.relu7 = fc_relu(n.fc6,3136)
    n.bn7,n.scale7 = batchnorm(n.fc7,stage=stage)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    #the following will be 7x7 (original /32).
    n.reshape8 = L.Reshape(n.fc7, reshape_param = dict(shape=dict(dim=[0,-1,7,7])))     # batchsize X infer X 7 X 7 , infer should=6272/49=128

    n.conv8_0,n.relu8_0,n.bn8_0,n.scale8_0 = conv_relu_bn(n.reshape8,n_output=512,kernel_size=7,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...

    #the following will be 14x14 (original /16).
    n.deconv8 = L.Deconvolution(n.conv8_0,
                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                            num_output=64,
                            convolution_param = dict(num_output=512, pad = 0,
                            kernel_size=2,
                            stride = 2,
                            weight_filler= {'type':'xavier'},
                            bias_filler= {'type':'constant','value':0.2}) )

#    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
#                                num_output=n_output, pad=pad, bias_term=False, weight_filler=dict(type='msra'))

    n.conv8_1,n.relu8_1,n.bn8_1,n.scale8_1 = conv_relu_bn(n.deconv8,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv8_cross1,n.relu8_cross1,n.bn8_cross1,n.scale8_cross1 = conv_relu_bn(n.conv5_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv8_cross2,n.relu8_cross2,n.bn8_cross2,n.scale8_cross2 = conv_relu_bn(n.conv8_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv8_cross2, n.conv8_1]
    n.cat8 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv8_2,n.relu8_2 = conv_relu(n.cat8,n_output=512,kernel_size=5,pad='preserve')


#yes 9 is missing, sue me

#the following will be 28x28  (original /16)
#deconv doesnt work from python , so these need to be changed by hand #
    # this is the 'more eff
    # icient equivalent' as listed in fb paper, except with extra relu's . try strict rewrite if this doesnt work

    n.deconv10 = L.Deconvolution(n.conv8_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=512,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))


    n.conv10_1,n.relu10_1,n.bn10_1,n.scale10_1 = conv_relu_bn(n.deconv10,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
#    n.conv10_2 = conv_bn_relu(n.conv10_1,n_output=512,kernel_size=3,pad='preserve')  #watch out for padsize here, make sure outsize is 14x14 #indeed
    n.conv4_cross1,n.relu4_cross1,n.bn4_cross1,n.scale4_cross1 = conv_relu_bn(n.conv4_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_cross2,n.relu4_cross2,n.bn4_cross2,n.scale4_cross2 = conv_relu_bn(n.conv4_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv4_cross2, n.conv10_1]
    n.cat10 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv10_2,n.relu10_2 = conv_relu(n.cat10,n_output=512,kernel_size=3,pad='preserve')

    #the following will be 56x56  (original /4)
    n.deconv11 = L.Deconvolution(n.conv10_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=256,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv11_1,n.relu11_1,n.bn11_1,n.scale11_1 = conv_relu_bn(n.deconv11,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.conv3_cross1,n.relu3_cross1,n.bn3_cross1,n.scale3_cross1 = conv_relu_bn(n.conv3_3,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_cross2,n.relu3_cross2,n.bn3_cross2,n.scale3_cross2 = conv_relu_bn(n.conv3_cross1,n_output=256,kernel_size=3,pad='preserve')
    bottom=[n.conv3_cross2, n.conv11_1]
    n.cat11 = L.Concat(*bottom)
    n.conv11_2,n.relu11_2,n.bn11_2,n.scale11_2 = conv_relu_bn(n.cat11,n_output=256,kernel_size=3,pad='preserve')

    #the following will be 112x112  (original /4)
    n.deconv12 = L.Deconvolution(n.conv11_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=128,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv12_1,n.relu12_1,n.bn12_1,n.scale12_1 = conv_relu_bn(n.deconv12,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.conv2_cross1,n.relu2_cross1,n.bn2_cross1,n.scale2_cross1 = conv_relu_bn(n.conv2_2,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_cross2,n.relu2_cross2,n.bn2_cross2,n.scale2_cross2 = conv_relu_bn(n.conv2_cross1,n_output=128,kernel_size=3,pad='preserve')
    bottom=[n.conv2_cross2, n.conv12_1]
    n.cat12 = L.Concat(*bottom)
    n.conv12_2,n.relu12_2,n.bn12_2,n.scale12_2 = conv_relu_bn(n.cat12,n_output=128,kernel_size=3,pad='preserve')


    #the following will be 224x224  (original /2)
    n.deconv13 = L.Deconvolution(n.conv12_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=64,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv13_1,n.relu13_1,n.bn13_1,n.scale13_1 = conv_relu_bn(n.deconv13,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.conv1_cross1,n.relu1_cross1,n.bn1_cross1,n.scale1_cross1 = conv_relu_bn(n.conv1_2,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_cross2,n.relu1_cross2,n.bn1_cross2,n.scale1_cross2 = conv_relu_bn(n.conv1_cross1,n_output=64,kernel_size=3,pad='preserve')
    bottom=[n.conv1_cross2, n.conv13_1]


    n.cat13 = L.Concat(*bottom)
    n.conv13_2,n.relu13_2,n.bn13_2,n.scale13_2 = conv_relu_bn(n.cat13,n_output=64,kernel_size=3,pad='preserve',stage=stage)  #this is halving N_filters

    n.conv_final = conv(n.conv13_2,n_output=n_cats,kernel_size=3,pad='preserve')

#    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label,normalize=True)
    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label)

#    n.deconv1 = L.Deconvolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                convolution_param=[dict(num_output=512,bias_term=False,kernel_size=2,stride=2)])
    return n.to_proto()

def sharp5(db,mean_value=[112.0,112.0,112.0],imsize=(224,224),n_cats=21,stage='train'):
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
    orig_size = imsize[0]
    pool_count=0
    n=caffe.NetSpec()
    #assuming input of size 224x224, ...
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=mean_value,mirror=True),ntop=2)

    n.bn1,n.scale1 = batchnorm(n.data,stage=stage)
    n.conv1_1,n.relu1_1,n.bn1_1,n.scale1_1 = conv_relu_bn(n.scale1,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_2,n.relu1_2,n.bn1_2,n.scale1_2 = conv_relu_bn(n.conv1_1,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 112x112   (/2)
    n.conv2_1,n.relu2_1,n.bn2_1,n.scale2_1 = conv_relu_bn(n.pool1,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_2,n.relu2_2,n.bn2_2,n.scale2_2 = conv_relu_bn(n.conv2_1,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 56x56  (/4)
    n.conv3_1,n.relu3_1,n.bn3_1,n.scale3_1 = conv_relu_bn(n.pool2,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_2,n.relu3_2,n.bn3_2,n.scale3_2 = conv_relu_bn(n.conv3_1,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_3,n.relu3_3,n.bn3_3,n.scale3_3 = conv_relu_bn(n.conv3_2,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 28x28  (/8)
    n.conv4_1,n.relu4_1,n.bn4_1,n.scale4_1 = conv_relu_bn(n.pool3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_2,n.relu4_2,n.bn4_2,n.scale4_2 = conv_relu_bn(n.conv4_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_3,n.relu4_3,n.bn4_3,n.scale4_3 = conv_relu_bn(n.conv4_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 14x14  (/16)
    n.conv5_1,n.relu5_1,n.bn5_1,n.scale5_1 = conv_relu_bn(n.pool4,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_2,n.relu5_2,n.bn5_2,n.scale5_2 = conv_relu_bn(n.conv5_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_3,n.relu5_3,n.bn5_3,n.scale2_3 = conv_relu_bn(n.conv5_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be   /32
    n.fc6,n.relu6 = fc_relu(n.pool5,3136)  #6272=7*7*128
    n.bn6,n.scale6 = batchnorm(n.fc6,stage=stage)
    n.drop6 = L.Dropout(n.fc6, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    n.fc7,n.relu7 = fc_relu(n.fc6,3136)
    n.bn7,n.scale7 = batchnorm(n.fc7,stage=stage)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    current_size=orig_size/(2**pool_count)
    print("innermost size:"+str(current_size))
    n.reshape8 = L.Reshape(n.fc7, reshape_param = dict(shape=dict(dim=[0,-1,current_size,current_size])))     # batchsize X infer X 7 X 7 , infer should=6272/49=128

    n.conv8_0,n.relu8_0,n.bn8_0,n.scale8_0 = conv_relu_bn(n.reshape8,n_output=512,kernel_size=7,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...

    #the following will be 14x14 (original /16).
    n.deconv8 = L.Deconvolution(n.conv8_0,
                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                            num_output=64,
                            convolution_param = dict(num_output=512, pad = 0,
                            kernel_size=2,
                            stride = 2,
                            weight_filler= {'type':'xavier'},
                            bias_filler= {'type':'constant','value':0.2}) )

    n.conv8_1,n.relu8_1,n.bn8_1,n.scale8_1 = conv_relu_bn(n.deconv8,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv8_cross1,n.relu8_cross1,n.bn8_cross_1,n.scale8_cross1 = conv_relu_bn(n.conv5_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv8_cross2,n.relu8_cross2,n.bn8_cross_2,n.scale8_cross2 = conv_relu_bn(n.conv8_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv8_cross2, n.conv8_1]
    n.cat8 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv8_2,n.relu8_2 = conv_relu(n.cat8,n_output=512,kernel_size=5,pad='preserve')

    n.deconv10 = L.Deconvolution(n.conv8_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=512,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))

    n.conv10_1,n.relu10_1,n.bn10_1,n.scale10_1 = conv_relu_bn(n.deconv10,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
#    n.conv10_2 = conv_bn_relu(n.conv10_1,n_output=512,kernel_size=3,pad='preserve')  #watch out for padsize here, make sure outsize is 14x14 #indeed
    n.conv4_cross1,n.relu4_cross1,n.bn4_cross_1,n.scale4_cross1= conv_relu_bn(n.conv4_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_cross2,n.relu4_cross2,n.bn4_cross_2,n.scale4_cross2 = conv_relu_bn(n.conv4_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv4_cross2, n.conv10_1]
    n.cat10 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv10_2,n.relu10_2,n.bn10_2,n.scale10_2 = conv_relu_bn(n.cat10,n_output=512,kernel_size=3,pad='preserve')

    #the following will be 56x56  (original /4)
    n.deconv11 = L.Deconvolution(n.conv10_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=256,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv11_1,n.relu11_1,n.bn11_1,n.scale11_1 = conv_relu_bn(n.deconv11,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.conv3_cross1,n.relu3_cross1,n.bn3_cross_1,n.scale3_cross1 = conv_relu_bn(n.conv3_3,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_cross2,n.relu3_cross2,n.bn3_cross_2,n.scale3_cross2 = conv_relu_bn(n.conv3_cross1,n_output=256,kernel_size=3,pad='preserve')
    bottom=[n.conv3_cross2, n.conv11_1]
    n.cat11 = L.Concat(*bottom)
    n.conv11_2,n.relu11_2,n.bn11_2,n.scale11_2 = conv_relu_bn(n.cat11,n_output=256,kernel_size=3,pad='preserve')

    #the following will be 112x112  (original /4)
    n.deconv12 = L.Deconvolution(n.conv11_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=128,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv12_1,n.relu12_1,n.bn12_1,n.scale12_1 = conv_relu_bn(n.deconv12,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.conv2_cross1,n.relu2_cross1,n.bn2_cross_1,n.scale2_cross1 = conv_relu_bn(n.conv2_2,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_cross2,n.relu2_cross2,n.bn2_cross_2,n.scale2_cross2 = conv_relu_bn(n.conv2_cross1,n_output=128,kernel_size=3,pad='preserve')
    bottom=[n.conv2_cross2, n.conv12_1]
    n.cat12 = L.Concat(*bottom)
    n.conv12_2,n.relu12_2,n.bn12_2,n.scale12_2 = conv_relu_bn(n.cat12,n_output=128,kernel_size=3,pad='preserve')

    #the following will be 224x224  (original /2)
    n.deconv13 = L.Deconvolution(n.conv12_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=64,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv13_1,n.relu13_1,n.bn13_1,n.scale13_1 = conv_relu_bn(n.deconv13,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.conv1_cross1,n.relu1_cross1,n.bn1_cross_1,n.scale1_cross1 = conv_relu_bn(n.conv1_2,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_cross2,n.relu1_cross2,n.bn1_cross_2,n.scale1_cross2 = conv_relu_bn(n.conv1_cross1,n_output=64,kernel_size=3,pad='preserve')
    bottom=[n.conv1_cross2, n.conv13_1]
    n.cat13 = L.Concat(*bottom)
    n.conv13_2,n.relu13_2,n.bn13_2,n.scale13_2 = conv_relu_bn(n.cat13,n_output=64,kernel_size=3,pad='preserve',stage=stage)  #this is halving N_filters

    n.conv_final = conv(n.conv13_2,n_output=n_cats,kernel_size=3,pad='preserve')

#    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label,normalize=True)
    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label)

#    n.deconv1 = L.Deconvolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                convolution_param=[dict(num_output=512,bias_term=False,kernel_size=2,stride=2)])
    return n.to_proto()

def sharp6(db,mean_value=[112.0,112.0,112.0],imsize=(224,224),n_cats=21,stage='train'):
    '''
    see https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt
    like sharp5 but deeper
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
    orig_size = imsize[0]
    pool_count=0
    n=caffe.NetSpec()
    #assuming input of size 224x224, ...
    n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=db,transform_param=dict(scale=1./255,mean_value=mean_value,mirror=True),ntop=2)

    n.bn1,n.scale1 = batchnorm(n.data,stage=stage)
    n.conv1_0,n.relu1_0,n.bn1_0,n.scale1_0 = conv_relu_bn(n.scale1,n_output=64,kernel_size=5,pad='preserve')
    n.conv1_1,n.relu1_1,n.bn1_1,n.scale1_1 = conv_relu_bn(n.conv1_0,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_2,n.relu1_2,n.bn1_2,n.scale1_2 = conv_relu_bn(n.conv1_1,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.pool1 = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 112x112   (/2)
    n.conv2_0,n.relu2_0,n.bn2_0,n.scale2_0 = conv_relu_bn(n.pool1,n_output=128,kernel_size=5,pad='preserve')
    n.conv2_1,n.relu2_1,n.bn2_1,n.scale2_1 = conv_relu_bn(n.conv2_0,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_2,n.relu2_2,n.bn2_2,n.scale2_2 = conv_relu_bn(n.conv2_1,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 56x56  (/4)
    n.conv3_0,n.relu3_0,n.bn3_0,n.scale3_0 = conv_relu_bn(n.pool2,n_output=256,kernel_size=5,pad='preserve')
    n.conv3_1,n.relu3_1,n.bn3_1,n.scale3_1 = conv_relu_bn(n.conv3_0,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_2,n.relu3_2,n.bn3_2,n.scale3_2 = conv_relu_bn(n.conv3_1,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_3,n.relu3_3,n.bn3_3,n.scale3_3 = conv_relu_bn(n.conv3_2,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.pool3 = L.Pooling(n.conv3_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 28x28  (/8)
    n.conv4_0,n.relu4_0,n.bn4_0,n.scale4_0 = conv_relu_bn(n.pool3,n_output=512,kernel_size=5,pad='preserve')
    n.conv4_1,n.relu4_1,n.bn4_1,n.scale4_1 = conv_relu_bn(n.conv4_0,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_2,n.relu4_2,n.bn4_2,n.scale4_2 = conv_relu_bn(n.conv4_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_3,n.relu4_3,n.bn4_3,n.scale4_3 = conv_relu_bn(n.conv4_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool4 = L.Pooling(n.conv4_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be 14x14  (/16)
    n.conv5_0,n.relu5_0,n.bn5_0,n.scale5_0 = conv_relu_bn(n.pool4,n_output=512,kernel_size=5,pad='preserve')
    n.conv5_1,n.relu5_1,n.bn5_1,n.scale5_1 = conv_relu_bn(n.conv5_0,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_2,n.relu5_2,n.bn5_2,n.scale5_2 = conv_relu_bn(n.conv5_1,n_output=512,kernel_size=3,pad='preserve')
    n.conv5_3,n.relu5_3,n.bn5_3,n.scale2_3 = conv_relu_bn(n.conv5_2,n_output=512,kernel_size=3,pad='preserve',stage=stage)
    n.pool5 = L.Pooling(n.conv5_3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    pool_count+=1

    #the following will be   /32
    n.fc6,n.relu6 = fc_relu(n.pool5,3136)  #6272=7*7*128
    n.bn6,n.scale6 = batchnorm(n.fc6,stage=stage)
    n.drop6 = L.Dropout(n.fc6, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    n.fc7,n.relu7 = fc_relu(n.fc6,3136)
    n.bn7,n.scale7 = batchnorm(n.fc7,stage=stage)
    n.drop7 = L.Dropout(n.fc7, dropout_param=dict(dropout_ratio=0.2),in_place=True)

    current_size=orig_size/(2**pool_count)
    print("innermost size:"+str(current_size))
    n.reshape8 = L.Reshape(n.fc7, reshape_param = dict(shape=dict(dim=[0,-1,current_size,current_size])))     # batchsize X infer X 7 X 7 , infer should=6272/49=128

    n.conv8_0,n.relu8_0,n.bn8_0,n.scale8_0 = conv_relu_bn(n.reshape8,n_output=512,kernel_size=7,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...

    #the following will be 14x14 (original /16).
    n.deconv8 = L.Deconvolution(n.conv8_0,
                            param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                            num_output=64,
                            convolution_param = dict(num_output=512, pad = 0,
                            kernel_size=2,
                            stride = 2,
                            weight_filler= {'type':'xavier'},
                            bias_filler= {'type':'constant','value':0.2}) )

    n.conv8_0,n.relu8_0,n.bn8_0,n.scale8_0 = conv_relu_bn(n.deconv8,n_output=512,kernel_size=5,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv8_1,n.relu8_1,n.bn8_1,n.scale8_1 = conv_relu_bn(n.conv8_0,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv8_cross1,n.relu8_cross1,n.bn8_cross_1,n.scale8_cross1 = conv_relu_bn(n.conv5_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv8_cross2,n.relu8_cross2,n.bn8_cross_2,n.scale8_cross2 = conv_relu_bn(n.conv8_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv8_cross2, n.conv8_1]
    n.cat8 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv8_2,n.relu8_2 = conv_relu(n.cat8,n_output=512,kernel_size=5,pad='preserve')

    n.deconv10 = L.Deconvolution(n.conv8_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=512,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))

    n.conv10_0,n.relu10_0,n.bn10_0,n.scale10_0 = conv_relu_bn(n.deconv10,n_output=512,kernel_size=5,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv10_1,n.relu10_1,n.bn10_1,n.scale10_1 = conv_relu_bn(n.conv10_0,n_output=512,kernel_size=3,pad='preserve',stage=stage)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
#    n.conv10_2 = conv_bn_relu(n.conv10_1,n_output=512,kernel_size=3,pad='preserve')  #watch out for padsize here, make sure outsize is 14x14 #indeed
    n.conv4_cross1,n.relu4_cross1,n.bn4_cross_1,n.scale4_cross1= conv_relu_bn(n.conv4_3,n_output=512,kernel_size=3,pad='preserve')
    n.conv4_cross2,n.relu4_cross2,n.bn4_cross_2,n.scale4_cross2 = conv_relu_bn(n.conv4_cross1,n_output=512,kernel_size=3,pad='preserve')

    bottom = [n.conv4_cross2, n.conv10_1]
    n.cat10 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv10_2,n.relu10_2,n.bn10_2,n.scale10_2 = conv_relu_bn(n.cat10,n_output=512,kernel_size=3,pad='preserve')

    #the following will be 56x56  (original /4)
    n.deconv11 = L.Deconvolution(n.conv10_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=256,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))

    n.conv11_0,n.relu11_0,n.bn11_0,n.scale11_0 = conv_relu_bn(n.deconv11,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.conv11_1,n.relu11_1,n.bn11_1,n.scale11_1 = conv_relu_bn(n.conv11_0,n_output=256,kernel_size=3,pad='preserve',stage=stage)
    n.conv3_cross1,n.relu3_cross1,n.bn3_cross_1,n.scale3_cross1 = conv_relu_bn(n.conv3_3,n_output=256,kernel_size=3,pad='preserve')
    n.conv3_cross2,n.relu3_cross2,n.bn3_cross_2,n.scale3_cross2 = conv_relu_bn(n.conv3_cross1,n_output=256,kernel_size=3,pad='preserve')
    bottom=[n.conv3_cross2, n.conv11_1]
    n.cat11 = L.Concat(*bottom)
    n.conv11_2,n.relu11_2,n.bn11_2,n.scale11_2 = conv_relu_bn(n.cat11,n_output=256,kernel_size=3,pad='preserve')

    #the following will be 112x112  (original /4)
    n.deconv12 = L.Deconvolution(n.conv11_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=128,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv12_0,n.relu12_0,n.bn12_0,n.scale12_0 = conv_relu_bn(n.deconv12,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.conv12_1,n.relu12_1,n.bn12_1,n.scale12_1 = conv_relu_bn(n.conv12_0,n_output=128,kernel_size=3,pad='preserve',stage=stage)
    n.conv2_cross1,n.relu2_cross1,n.bn2_cross_1,n.scale2_cross1 = conv_relu_bn(n.conv2_2,n_output=128,kernel_size=3,pad='preserve')
    n.conv2_cross2,n.relu2_cross2,n.bn2_cross_2,n.scale2_cross2 = conv_relu_bn(n.conv2_cross1,n_output=128,kernel_size=3,pad='preserve')
    bottom=[n.conv2_cross2, n.conv12_1]
    n.cat12 = L.Concat(*bottom)
    n.conv12_2,n.relu12_2,n.bn12_2,n.scale12_2 = conv_relu_bn(n.cat12,n_output=128,kernel_size=3,pad='preserve')

    #the following will be 224x224  (original /2)
    n.deconv13 = L.Deconvolution(n.conv12_2,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    convolution_param = dict(num_output=64,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2)))
    n.conv13_0,n.relu13_0,n.bn13_0,n.scale13_0 = conv_relu_bn(n.deconv13,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.conv13_1,n.relu13_1,n.bn13_1,n.scale13_1 = conv_relu_bn(n.conv13_0,n_output=64,kernel_size=3,pad='preserve',stage=stage)
    n.conv1_cross1,n.relu1_cross1,n.bn1_cross_1,n.scale1_cross1 = conv_relu_bn(n.conv1_2,n_output=64,kernel_size=3,pad='preserve')
    n.conv1_cross2,n.relu1_cross2,n.bn1_cross_2,n.scale1_cross2 = conv_relu_bn(n.conv1_cross1,n_output=64,kernel_size=3,pad='preserve')
    bottom=[n.conv1_cross2, n.conv13_1]
    n.cat13 = L.Concat(*bottom)
    n.conv13_2,n.relu13_2,n.bn13_2,n.scale13_2 = conv_relu_bn(n.cat13,n_output=64,kernel_size=3,pad='preserve',stage=stage)  #this is halving N_filters

    n.conv_final = conv(n.conv13_2,n_output=n_cats,kernel_size=3,pad='preserve')

#    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label,normalize=True)
    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label)

#    n.deconv1 = L.Deconvolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                convolution_param=[dict(num_output=512,bias_term=False,kernel_size=2,stride=2)])
    return n.to_proto()

def sharp_res50():
    pass

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

def unet(db,mean_value=[112.0,112.0,112.0],n_cats=21):
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

    n.conv7_1,n.relu7_1 = conv_relu(n.deconv7,n_output=512,kernel_size=2,pad=0)  #watch out for padsize here, make sure outsize is 14x14 #ug, pad1->size15, pad0->size13...
    n.conv7_1,n.relu7_1 = conv_relu(n.deconv7,n_output=512,kernel_size=3,pad=1)  #watch out for padsize here, make sure outsize is 14x14 #indeed
    bottom=[n.conv5_3, n.conv7_1]
    n.cat7 = L.Concat(*bottom) #param=dict(concat_dim=1))
    n.conv7_2,n.relu7_2 = conv_relu(n.cat7,n_output=1024,kernel_size=3,pad=1)
    n.conv7_3,n.relu7_3 = conv_relu(n.conv7_2,n_output=1024,kernel_size=3,pad=1)

    #the following will be 28x28  (original /8)
    n.deconv8 = L.Convolution(n.conv7_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],#
                    num_output=1024,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv8_1,n.relu8_1 = conv_relu(n.deconv8,n_output=512,kernel_size=3,pad=1)
    bottom=[n.conv4_3, n.conv8_1]
    n.cat8 = L.Concat(*bottom)
    n.conv8_2,n.relu8_2 = conv_relu(n.cat8,n_output=512,kernel_size=3,pad=1)  #this is halving N_filters
    n.conv8_3,n.relu8_3 = conv_relu(n.conv8_2,n_output=512,kernel_size=3,pad=1)

    #the following will be 56x56  (original /4)
    n.deconv9 = L.Convolution(n.conv8_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    num_output=512,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv9_1,n.relu9_1 = conv_relu(n.deconv9,n_output=256,kernel_size=3,pad=1)
    bottom=[n.conv3_3, n.conv9_1]
    n.cat9 = L.Concat(*bottom)
    n.conv9_2,n.relu9_2 = conv_relu(n.cat9,n_output=256,kernel_size=3,pad=1)  #this is halving N_filters
    n.conv9_3,n.relu9_3 = conv_relu(n.conv9_2,n_output=256,kernel_size=3,pad=1)

    #the following will be 112x112  (original /2)
    n.deconv10 = L.Convolution(n.conv9_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    num_output=256,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv10_1,n.relu10_1 = conv_relu(n.deconv10,n_output=128,kernel_size=3,pad=1)
    bottom=[n.conv2_2, n.conv10_1]
    n.cat10 = L.Concat(*bottom)
    n.conv10_2,n.relu10_2 = conv_relu(n.cat10,n_output=128,kernel_size=3,pad=1)  #this is halving N_filters
    n.conv10_3,n.relu10_3 = conv_relu(n.conv10_2,n_output=128,kernel_size=3,pad=1)

    #the following will be 224x224  (original)
    n.deconv11 = L.Convolution(n.conv10_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                    num_output=128,pad = 0,kernel_size=2,stride = 2,
                    weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))
    n.conv11_1,n.relu11_1 = conv_relu(n.deconv11,n_output=64,kernel_size=3,pad=1)
    bottom=[n.conv1_2, n.conv11_1]
    n.cat11 = L.Concat(*bottom)
    n.conv11_2,n.relu11_2 = conv_relu(n.cat11,n_output=64,kernel_size=3,pad=1)  #this is halving N_filters
    n.conv11_3,n.relu11_3 = conv_relu(n.conv11_2,n_output=64,kernel_size=3,pad=1)

    n.conv_final,n.relu_final = conv_relu(n.conv11_3,n_output=n_cats,kernel_size=3,pad=1)

#    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label,normalize=True)
    n.loss = L.SoftmaxWithLoss(n.conv_final, n.label)

#    n.deconv1 = L.Deconvolution(n.conv6_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
#                convolution_param=[dict(num_output=512,bias_term=False,kernel_size=2,stride=2)])
    return n.to_proto()



''' #







    #the following will be 112x112  (original /2)
    n.deconv3 = L.Convolution(n.conv9_3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=512,pad = 0,kernel_size=2,stride = 2,
                            weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))

    #the following will be 224x224  (original /1)
    n.deconv4 = L.Convolution(n.deconv3,param=[dict(lr_mult=lr_mult1,decay_mult=decay_mult1),dict(lr_mult=lr_mult2,decay_mult=decay_mult2)],
                            num_output=512,pad = 0,kernel_size=2,stride = 2,
                            weight_filler=dict(type='xavier'),bias_filler=dict(type='constant',value=0.2))





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

def estimate_mem(prototxt):
    caffe.set_mode_gpu()

    solver = caffe.SGDSolver(prototxt)
    #solver.net.copy_from(weights)
    #solver.net.forward()  # train net  #doesnt do fwd and backwd passes apparently

    # surgeries
    #interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    all_layers = [k for k in solver.net.params.keys()]
    print('all layers:')
    print all_layers

    for k,v in solver.net.params:
        print('key {} val {}'.format(k,v))

    for k,v in solver.net.blobs:
        print('key {} val {}'.format(k,v))


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
        outstring = outstring+line+'\n'
    return outstring

def replace_pythonlayer(proto,stage='train'):
    '''the built in stuff doesnt appear to be able to handle a custom python layer
    so here i replcae by hand
    '''
    pythonlayer = 'layer {\n    name: \"data\"\n    type: \"Python\"\n    top: \"data\"\n    top: \"label\"\n    python_param {\n    module: \"jrlayers2\"\n    layer: \"JrPixlevel\"\n    param_str: \"{\\\"images_and_labels_file\\\": \\\"/data/jeremy/image_dbs/pixlevel/pixlevel_fullsize_train_labels_v3.txt\\\", \\\"mean\\\": (104.0, 116.7, 122.7),\\\"augment\\\":True,\\\"resize\\\":(300,300),\\\"augment_crop_size\\\":(256,256), \\\"batch_size\\\":9 }\"\n    }\n  }\n'
    if stage == 'test':
        pythonlayer = 'layer {\n    name: \"data\"\n    type: \"Python\"\n    top: \"data\"\n    top: \"label\"\n    python_param {\n    module: \"jrlayers2\"\n    layer: \"JrPixlevel\"\n    param_str: \"{\\\"images_and_labels_file\\\": \\\"/data/jeremy/image_dbs/pixlevel/pixlevel_fullsize_test_labels_v3.txt\\\", \\\"mean\\\": (104.0, 116.7, 122.7),\\\"augment\\\":True,\\\"resize\\\":(300,300),\\\"augment_crop_size\\\":(256,256), \\\"batch_size\\\":1 }\"\n    }\n  }\n'
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
#            print('layer buf:')
#            print layer_buf
            first_layer = False
            outstring = outstring + layer_buf
            layer_buf = 'layer {\n'
        if new_layer_flag and first_layer:
            first_layer = False
    #dont forget the final layer
    outstring = outstring + layer_buf

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

  #  estimate_mem('val.prototxt')

    print('starting to generate net')
#    proto = vgg16('thedb')
#    proto = unet('thedb')
#    proto = test_convbnrelu('thedb')
#    proto = correct_deconv(str(proto))

#sharp2
    # proto = sharp2('thedb',stage='train')
    # proto = replace_pythonlayer(str(proto),stage='train')
    # with open('train.prototxt','w') as f:
    #     f.write(str(proto))
    #     f.close()
    #
    # proto = sharp2('thedb',stage='test')
    # proto = replace_pythonlayer(str(proto),stage='test')
    # with open('val.prototxt','w') as f:
    #     f.write(str(proto))
    #     f.close()

#sharp6
    # proto = sharp6('thedb',stage='train')
    # proto = replace_pythonlayer(str(proto),stage='train')
    # with open('s6_train.prototxt','w') as f:
    #     f.write(str(proto))
    #     f.close()
    #
    # proto = sharp6('thedb',stage='test')
    # proto = replace_pythonlayer(str(proto),stage='test')
    # with open('s6_val.prototxt','w') as f:
    #     f.write(str(proto))
    #     f.close()


#resU
    proto = jr_resnet_u()
    proto = replace_pythonlayer(str(proto),stage='train')
    with open('train.prototxt','w') as f:
        f.write(str(proto))
        f.close()
    proto = jr_resnet_u()
    proto = replace_pythonlayer(str(proto),stage='test')
    with open('val.prototxt','w') as f:
        f.write(str(proto))
        f.close()

#    estimate_mem('val.prototxt')

#    caffe.set_device(2)
#    caffe.set_mode_gpu()
#    solver = caffe.SGDSolver('solver.prototxt')
#    weights = 'snapshot/train_0816__iter_25000.caffemodel'  #in brainia container jr2
#    solver.net.copy_from(weights)

    # surgeries
 #   interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#    all_layers = [k for k in solver.net.params.keys()]
#    surgery.interp(solver.net, interp_layers)
    # scoring
    #val = np.loadtxt('../data/segvalid11.txt', dtype=str)
#    val = range(0,1500)
    #jrinfer.seg_tests(solver, False, val, layer='score')

#    for _ in range(1000):
#        solver.step(1)
