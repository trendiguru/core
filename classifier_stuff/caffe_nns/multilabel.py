__author__ = 'jeremy' #ripped from tutorial at http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/pascal-multilabel-with-datalayer.ipynb


import sys
import os

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from copy import copy
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.


# matplotlib inline
def setup():
    lt.rcParams['figure.figsize'] = (6, 6)

    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    sys.path.append(caffe_root + 'python')
    sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
    sys.path.append("pycaffe") # the tools file is in this folder

    import tools #this contains some tools that we need

    # set data root directory, e.g:
    pascal_root = osp.join(caffe_root, 'data/pascal/VOC2012')

    # these are the PASCAL classes, we'll need them later.
    classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])

    # make sure we have the caffenet weight downloaded.
    if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print("Downloading pre-trained CaffeNet model...")
    #    !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet


# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer):
    # setup the python data layer
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'pascal_multilabel_datalayers', layer = datalayer,
                               ntop = 2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=20)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)

    return str(n.to_proto())

def makenet():
    workdir = './pascal_multilabel_with_datalayer'
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
    solverprototxt.sp['display'] = "1"
    solverprototxt.sp['base_lr'] = "0.0001"
    solverprototxt.write(osp.join(workdir, 'solver.prototxt'))

    # write train net.
    with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
        # provide parameters to the data layer as a python dictionary. Easy as pie!
        data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train', pascal_root = pascal_root)
        f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))

    # write validation net.
    with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
        data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'val', pascal_root = pascal_root)
        f.write(caffenet_multilabel(data_layer_params, 'PascalMultilabelDataLayerSync'))

    solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
    solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    solver.test_nets[0].share_with(solver.net)
    solver.step(1)


    ## check images loaded by batchloader
    transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
    image_index = 0 # First image in the batch.
    plt.figure()
    plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
    gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
    plt.title('GT: {}'.format(classes[np.where(gtlist)]))
    plt.axis('off');


def hamming_distance(gt, est):
    #this is actually hamming similarity not distance
    print('calculating hamming for \ngt :'+str(gt)+'\nest:'+str(est))
    if est.shape != gt.shape:
        est = est.reshape(gt.shape)
        print('after reshape:size gt {} size est {}'.format(gt.shape,est.shape))
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def update_confmat(gt,est,tp,tn,fp,fn):
    print('gt {} \nest {} sizes tp {} tn {} fp {} fn {} '.format(gt,est,tp.shape,tn.shape,fp.shape,fn.shape))
    for i in range(len(gt)):
        if gt[i] == 1:
            if est[i]: # true positive
                tp[i] += 1
            else:   # false negative
                fn[i] += 1
        else:
            if est[i]: # false positive
                fp[i] += 1
            else:   # true negative
                tn[i] += 1
#        print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    return tp,tn,fp,fn

def test_confmat():
    gt=[True,False,1,0]
    ests=[[True,False,0,0],
          [0,0,1,0],
          [1,0,0,1],
        [ True,0,True,0]]
    tp = [0,0,0,0]
    tn = [0,0,0,0]
    fp = [0,0,0,0]
    fn = [0,0,0,0]
    tp_sum = tn_sum = fp_sum = fn_sum = [0,0,0,0]
    for e in ests:
        #update_confmat(gt,e,tp,tn,fp,fn)
        tp,tn,fp,fn = update_confmat(gt,e,tp,tn,fp,fn)
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    gt=[0,1,1,0]
    ests=[[0,1,0,1],
          [0,1,1,1],
          [1,0,0,1],
          [1,0,1,0]]
    tp_sum = tn_sum = fp_sum = fn_sum = [0,0,0,0]
    for e in ests:
        #update_confmat(gt,e,tp,tn,fp,fn)
        tp,tn,fp,fn = update_confmat(gt,e,tp,tn,fp,fn)
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))



def check_acc(net, num_batches, batch_size = 1):
    #this is not working foir batchsize!=1, maybe needs to be defined in net
    acc = 0.0 #
    baseline_acc = 0.0
    n = 0

    first_time = True
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
#        ests = net.blobs['score'].data > 0  ##why 0????  this was previously not after a sigmoid apparently
        ests = net.blobs['score'].data > 0.5
        if first_time == True:
            first_time = False
            tp = np.zeros_like(gts)
            tn = np.zeros_like(gts)
            fp = np.zeros_like(gts)
            fn = np.zeros_like(gts)

        if ests.shape != gts.shape:
            ests = ests.reshape(gts.shape)
            print('after reshape in check_acc:size gt {} size est {}'.format(gts.shape,ests.shape))
        baseline_est = np.zeros_like(ests)
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            tp,tn,fp,fn = update_confmat(gt,est,tp,tn,fp,fn)
            h = hamming_distance(gt, est)

            baseline_h = hamming_distance(gt,baseline_est)
#            print('gt {} est {} (1-hamming) {}'.format(gt,est,h))
            sum = np.sum(gt)
            acc += h
            baseline_acc += baseline_h
            n += 1
    print('len(gts) {} len(ests) {} numbatches {} batchsize {} acc {} baseline {}'.format(len(gts),len(ests),num_batches,batch_size,acc/n,baseline_acc/n))
    print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
    full_rec = [float(tp[i])/(tp[i]+fn[i]) for i in range(len(tp))]
    full_prec = [float(tp[i])/(tp[i]+fp[i]) for i in range(len(tp))]
    full_acc = [float(tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i]) for i in range(len(tp))]
    print('precision {} recall {} acc {} nacc {}'.format(full_prec,full_rec,full_acc,acc/n))
    return acc / n

#train
def train():
    for itt in range(6):
        solver.step(100)
        print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 50))

def check_baseline_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, len(gts)))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            print('gt {} est {} '.format(gt,est))
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


def results():#prediction results
    test_net = solver.test_nets[0]
    for image_index in range(5):
        plt.figure()
        plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
        gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
        estlist = test_net.blobs['score'].data[image_index, ...] > 0
        plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
        plt.axis('off')


def check_accuracy(solverproto,caffemodel,num_batches=200,batch_size=1):
    solver = caffe.SGDSolver(solverproto)
    solver.net.copy_from(caffemodel)
    solver.test_nets[0].share_with(solver.net)
#    solver.step(1)
    print 'accuracy:{0:.4f}'.format(check_acc(solver.test_nets[0], num_batches=num_batches,batch_size = batch_size))



if __name__ =="__main__":
    caffe.set_mode_gpu()
    caffe.set_device(0)

    workdir = './'
    snapshot = 'snapshot'
    caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_40069.caffemodel'
    solverproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/solver.prototxt'
    check_accuracy(solverproto,caffemodel)
  #  print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 10,batch_size = 20))
