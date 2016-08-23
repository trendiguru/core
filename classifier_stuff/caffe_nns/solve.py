__author__ = 'jeremy'
#ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers.py /root/caffe/python
#ln -s /usr/lib/python2.7/dist-packacges/trendi/classifier_stuff/caffe_nns/surgery.py /root/caffe/python
#ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/score.py /root/caffe/python
#git -C /usr/lib/python2.7/dist-packages/trendi pull

import caffe
import surgery, score

import numpy as np
import os
import sys

import setproctitle
import subprocess
import socket

from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import progress_plot

setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '../vgg16fc.caffemodel'  #cannot find this
weights = 'snapshot/train_iter_1799.caffemodel'
weights = 'snapshot/train_iter_359310.caffemodel'
weights = 'snapshot/train_iter_370000.caffemodel'
weights = 'VGG_ILSVRC_16_layers.caffemodel'
weights = 'snapshot/train_0816__iter_25000.caffemodel'  #in brainia container jr2

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
all_layers = [k for k in solver.net.params.keys()]
print('all layers:')
print all_layers
#surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
val = range(0,1500)

#jrinfer.seg_tests(solver, False, val, layer='score')
hostname = socket.gethostname()
outfilename = hostname+'netoutput.txt'
jpgname = outfilename+'.jpg'
cmd = 'scp '+jpgname+' root@104.155.22.95:/var/www/results/progress_plots/';

jrinfer.seg_tests(solver, False, val, layer='score')
progress_plot.parse_solveoutput(outfilename)
subprocess.call(cmd,shell=True)

i = 0
losses = []
iters = []
for _ in range(1000):
    i = i+1
    solver.step(20)
    loss = solver.net.blobs['loss'].data
    print('loss:'+str(loss))
    losses.append(loss)
    iters.append(i)
#    score.seg_tests(solver, False, val, layer='score')
    jrinfer.seg_tests(solver, False, val, layer='score',outfilename=outfilename)
#    progress_plot.parse_solveoutput(outfilename)
    print('jpgfile:'+str(jpgname))
    subprocess.call(cmd,shell=True)
