__author__ = 'jeremy'
#ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/jrlayers.py /root/caffe/python
#ln -s /usr/lib/python2.7/dist-packacges/trendi/classifier_stuff/caffe_nns/surgery.py /root/caffe/python
#ln -s /usr/lib/python2.7/dist-packages/trendi/classifier_stuff/caffe_nns/score.py /root/caffe/python
#git -C /usr/lib/python2.7/dist-packages/trendi pull

import caffe
import surgery, score
import time
import numpy as np
import os
import sys

import setproctitle
import subprocess
import socket

import matplotlib
matplotlib.use('Agg') #allow plot generation on X-less systems
import matplotlib.pyplot as plt
plt.ioff()

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
solver.net.forward()  # train net

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
lossfilename = os.path.join('/home/jeremy/caffenets/production',hostname+'_loss.txt')
jpgname = outfilename+'.jpg'
cmd = 'scp '+jpgname+' root@104.155.22.95:/var/www/results/progress_plots/';
copycmd = 'cp '+jpgname +' /home/jeremy/caffenets/production'
copy2cmd = 'cp '+outfilename +' /home/jeremy/caffenets/production'

jrinfer.seg_tests(solver, False, val, layer='score')
progress_plot.parse_solveoutput(outfilename)
subprocess.call(cmd,shell=True)

i = 0
losses = []
iters = []
steps = 20
for _ in range(1000):
    for i in range(100):
        i = i+steps
        solver.step(steps)
        loss = solver.net.blobs['loss'].data
        print('iter '+str(i)+' loss:'+str(loss))
        losses.append(loss)
        iters.append(i)
        with open('loss.txt','a+') as f:
            f.write(str(int(time.time()))+'\t'+str(iter)+'\t'+str(loss)+'\n')
            f.close()
        with open(lossfilename,'a+') as f:
            f.write(str(int(time.time()))+'\t'+str(iter)+'\t'+str(loss)+'\n')
            f.close()

#    score.seg_tests(solver, False, val, layer='score')
    plt.plot(iters, loss,'bo:', label="train loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    savename = 'loss.jpg'
    plt.savefig(savename)
    jrinfer.seg_tests(solver, False, val, layer='score',outfilename=outfilename)
#    progress_plot.parse_solveoutput(outfilename)
    print('jpgfile:'+str(jpgname))
    subprocess.call(cmd,shell=True)
    subprocess.call(copycmd,shell=True)
    subprocess.call(copy2cmd,shell=True)
