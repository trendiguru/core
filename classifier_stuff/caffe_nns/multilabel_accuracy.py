__author__ = 'jeremy' #ripped from tutorial at http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/pascal-multilabel-with-datalayer.ipynb

import sys
import os

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import urllib2,urllib
from copy import copy
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import matplotlib.pyplot as plt

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import cv2


from trendi import constants
from trendi.utils import imutils



caffemodel =  '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
deployproto = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/deploy.prototxt'
caffe.set_mode_gpu()
caffe.set_device(0)
multilabel_net = caffe.Net(deployproto,caffemodel, caffe.TEST)


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
#    print('calculating hamming for \ngt :'+str(gt)+'\nest:'+str(est))
    if est.shape != gt.shape:
        print('shapes dont match')
        return 0
    hamming_similarity = sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))
    return hamming_similarity

def update_confmat(gt,est,tp,tn,fp,fn):
#    print('gt {} \nest {} sizes tp {} tn {} fp {} fn {} '.format(gt,est,tp.shape,tn.shape,fp.shape,fn.shape))
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

def check_acc(net, num_batches, batch_size = 1,threshold = 0.5):
    #this is not working foir batchsize!=1, maybe needs to be defined in net
    acc = 0.0 #
    baseline_acc = 0.0
    n = 0

    first_time = True
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
#        ests = net.blobs['score'].data > 0  ##why 0????  this was previously not after a sigmoid apparently
        ests = net.blobs['score'].data > threshold
        baseline_est = np.zeros_like(ests)
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if est.shape != gt.shape:
                print('shape mismatch')
                continue
            if first_time == True:
                first_time = False
                tp = np.zeros_like(gt)
                tn = np.zeros_like(gt)
                fp = np.zeros_like(gt)
                fn = np.zeros_like(gt)
            tp,tn,fp,fn = update_confmat(gt,est,tp,tn,fp,fn)
            print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))
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
    print('THRESHOLD '+str(threshold))
    print('precision {}\nrecall {}\nacc {}\navgacc {}'.format(full_prec,full_rec,full_acc,acc/n))
    return full_prec,full_rec,full_acc,tp,tn,fp,fn

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


def check_accuracy(proto,caffemodel,num_batches=200,batch_size=1,threshold = 0.5):
    print('checking accuracy of net {} using proto {}'.format(caffemodel,solverproto))
#    solver = caffe.SGDSolver(solverproto)
     # Make classifier.
    #classifier = caffe.Classifier(MODLE_FILE, PRETRAINED,
    #                          image_dims=image_dims, mean=mean,
    #                         input_scale=input_scale, raw_scale=raw_scale,
    #                          channel_swap=channel_swap)
    net = caffe.Net(proto,caffemodel, caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(1)

#    solver.net.copy_from(caffemodel)
#    solver.test_nets[0].share_with(solver.net)
#    solver.step(1)
#    precision,recall,accuracy,tp,tn,fp,fn = check_acc(solver.test_nets[0], num_batches=num_batches,batch_size = batch_size, threshold=threshold)
    precision,recall,accuracy,tp,tn,fp,fn = check_acc(net, num_batches=num_batches,batch_size = batch_size, threshold=threshold)
    return precision,recall,accuracy,tp,tn,fp,fn

def multilabel_infer_one(url):
    image_mean = np.array([104.0,117.0,123.0])
    input_scale = None
    channel_swap = [2, 1, 0]
    raw_scale = 255.0
    print('loading caffemodel for neurodoll (single class layers)')

    start_time = time.time()
    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)   #
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = net.blobs['siggy'].data[0][category_index] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out shape {}'.format(min,max,out.shape))
    out = out*255
    min = np.min(out)
    max = np.max(out)
    print('min {} max {} out after scaling  {}'.format(min,max,out.shape))
    result = Image.fromarray(out.astype(np.uint8))
#        outname = im.strip('.png')[0]+'out.bmp'
#    outname = os.path.basename(imagename)
#    outname = outname.split('.jpg')[0]+'.bmp'
#    outname = os.path.join(out_dir,outname)
#    print('outname:'+outname)
#    result.save(outname)
    #        fullout = net.blobs['score'].data[0]
    elapsed_time=time.time()-start_time
    print('infer_one elapsed time:'+str(elapsed_time))
 #   cv2.imshow('out',out.astype(np.uint8))
 #   cv2.waitKey(0)
    return out.astype(np.uint8)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    if url.count('jpg') > 1:
        return None

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    if image.size == 0:
        return None
    new_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return new_image

def get_multilabel_output(url_or_np_array,required_image_size=(227,227),output_layer_name='prob'):


    if isinstance(url_or_np_array, basestring):
        print('infer_one working on url:'+url_or_np_array)
        image = url_to_image(url_or_np_array)
    elif type(url_or_np_array) == np.ndarray:
        image = url_or_np_array
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
#    im = Image.open(imagename)
#    im = im.resize(required_imagesize,Image.ANTIALIAS)
#    in_ = in_.astype(float)
    in_ = imutils.resize_keep_aspect(image,output_size=required_image_size,output_file=None)
    in_ = np.array(in_, dtype=np.float32)   #.astype(float)
    if len(in_.shape) != 3:  #h x w x channels, will be 2 if only h x w
        print('got 1-chan image, turning into 3 channel')
        #DEBUG THIS , ORDER MAY BE WRONG [what order? what was i thinking???]
        in_ = np.array([in_,in_,in_])
    elif in_.shape[2] != 3:  #for rgb/bgr, some imgages have 4 chan for alpha i guess
        print('got n-chan image, skipping - shape:'+str(in_.shape))
        return
#    in_ = in_[:,:,::-1]  for doing RGB -> BGR : since this is loaded nby cv2 its unecessary
#    cv2.imshow('test',in_)
    in_ -= np.array((104,116,122.0))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    multilabel_net.blobs['data'].reshape(1, *in_.shape)
    multilabel_net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    multilabel_net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0) #for a parse with per-pixel max
    out = multilabel_net.blobs[output_layer_name].data[0] #for the nth class layer #siggy is after sigmoid
    min = np.min(out)
    max = np.max(out)
    print('out  {}'.format(out))




if __name__ =="__main__":
    #TODO dont use solver to get inferences , no need for solver for that
    caffe.set_mode_gpu()
    caffe.set_device(1)

    workdir = './'
    snapshot = 'snapshot'
#    caffemodel = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_240000.caffemodel'
#    caffemodel = '/home/jeremy/caffenets/multilabel/vgg_ilsvrc_16_multilabel_2/snapshot/train_iter_340000.caffemodel'
    caffemodel = '/home/jeremy/caffenets/production/multilabel_resnet50_sgd_iter_120000.caffemodel'
    model_base = caffemodel.split('/')[-1]
    solverproto = '/home/jeremy/caffenets/production/ResNet-50-test.prototxt'
    p_all = []
    r_all = []
    a_all = []
#    for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.92,0.95,0.98]:
    thresh = [0.5,0.9]
    for t in thresh:
        p,r,a,tp,tn,fp,fn = check_accuracy(solverproto, caffemodel, threshold=t, num_batches=800)
        p_all.append(p)
        r_all.append(p)
        a_all.append(p)
        with open('multilabel_accuracy_results.txt','a') as f:
            f.write('vgg_ilsvrc16_multilabel_2, threshold = '+str(t)+'\n')
            f.write('solver:'+solverproto+'\n')
            f.write('model:'+caffemodel+'\n')
            f.write('categories: '+str(constants.web_tool_categories)+ '\n')
            f.write('precision\n')
            f.write(str(p)+'\n')
            f.write('recall\n')
            f.write(str(r)+'\n')
            f.write('accuracy\n')
            f.write(str(a)+'\n')
            f.write('true positives\n')
            f.write(str(tp)+'\n')
            f.write('true negatives\n')
            f.write(str(tn)+'\n')
            f.write('false positives\n')
            f.write(str(fp)+'\n')
            f.write('false negatives\n')
            f.write(str(fn)+'\n')
    p_all_np = np.transpose(np.array(p_all))
    r_all_np = np.transpose(np.array(p_all))
    a_all_np = np.transpose(np.array(p_all))
    thresh_all_np = np.array(thresh)
    labels = constants.web_tool_categories
    plabels = [label + 'precision' for label in labels]
    rlabels = [label + 'recall' for label in labels]
    alabels = [label + 'accuracy' for label in labels]
    print('shape:'+str(p_all_np.shape))
    for i in range(p_all_np.shape[0]):
        plt.subplot(311)
        plt.plot(thresh_all_np,p_all_np[i,:],label=labels[i],linestyle='None')
        plt.subplot(312)   #
        plt.plot(thresh_all_np,r_all_np[i,:],label=labels[i],linestyle='None')
        plt.subplot(313)
        plt.plot(thresh_all_np,a_all_np[i,:],label=labels[i],linestyle='None')
    plt.subplot(311)
    plt.title('results '+model_base)
    plt.xlabel('threshold')
    plt.ylabel('precision')
    plt.subplot(312)   #
    plt.xlabel('threshold')
    plt.ylabel('recall')
    plt.subplot(313)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')

    plt.legend()
    plt.grid(True)
    plt.show()#
    plt.savefig('multilabel_results'+model_base+'.png', bbox_inches='tight')

  #  print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 10,batch_size = 20))
