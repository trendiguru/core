__author__ = 'jeremy'

import caffe
import time
import numpy as np
import os
import sys
import subprocess
import socket
from trendi import Utils


# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

weights = 'ResNet-50-model.caffemodel'

solver = caffe.SGDSolver('/home/yonatan/trendi/yonatan/resnet_50_dress_length/solver50_sgd.prototxt')
#solver.net.copy_from(weights)

docker_hostname = socket.gethostname()
host_dirname = '/home/yonatan/info_for_graphs' # the loss and accuracy will go here
Utils.ensure_dir(host_dirname) # if the folder doesn't exist - create it
baremetal_hostname = os.environ.get('HOST_HOSTNAME') # HOST_HOSTNAME = the server hostname (not the docker hostname)
prefix = baremetal_hostname+'.' + docker_hostname
results_outputname = prefix + 'results.txt'
copycmd = 'cp ' + results_outputname + ' ' + host_dirname
scpcmd = 'scp '+results_outputname+' root@104.155.22.95:/var/www/results/progress_plots/' # sending the info to extrimly

i = 0
steps_per_iter = 1 # not supposed to help if bigger than 1
n_iter = 100 # calculate loss every n_iter iterations
loss_avg = [0]*n_iter # create a list of length n_iter
accuracy_avg = [0]*n_iter # create a list of length n_iter
tot_iters = 0
for _ in range(100000):
    for i in range(n_iter):
        solver.step(steps_per_iter)
        loss = solver.net.blobs['loss'].data
        accuracy = solver.net.blobs['accuracy'].data
        print('iter '+str(i*steps_per_iter)+' loss:'+str(loss)+' accuracy:'+str(accuracy))
        loss_avg[i] = loss
        accuracy_avg[i] = accuracy
    tot_iters += steps_per_iter*n_iter
    averaged_loss=sum(loss_avg)/len(loss_avg)
    averaged_accuracy = sum(accuracy_avg) / len(accuracy_avg)
    with open(results_outputname,'a+') as f:
        f.write(str(int(time.time()))+'\t'+str(tot_iters)+'\t'+str(averaged_loss)+'\t'+str(averaged_accuracy)+'\n')
        f.close()

    subprocess.call(copycmd,shell=True) # run shell comands

    subprocess.call(scpcmd,shell=True) # run shell comands



