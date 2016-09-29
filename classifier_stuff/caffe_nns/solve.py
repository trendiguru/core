__author__ = 'jeremy'
import caffe
import time
import os
import sys
import setproctitle
import subprocess
import socket
import matplotlib
matplotlib.use('Agg') #allow plot generation on X-less systems
import matplotlib.pyplot as plt
plt.ioff()

from trendi import Utils
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy

setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'snapshot/train_0816__iter_25000.caffemodel'  #in brainia container jr2

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()
solverproto = 'solver.prototxt'
testproto = 'train_test.prototxt'

#solver = caffe.SGDSolver('solver.prototxt')
#get_solver is more general, SGDSolver forces sgd even if something else is specified in prototxt
solver = caffe.get_solver(solverproto)
training_net = solver.net
test_net = solver.test_nets[0] # more than one testnet is supported


#solver.net.copy_from(weights)
#solver.net.forward()  # train net  #doesnt do fwd and backwd passes apparently
# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
all_params = [k for k in solver.net.params.keys()]
print('all params:')
print all_params
all_blobs = [k for k in solver.net.blobs.keys()]
print('all blobs:')
print all_blobs
#surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
val = range(0,200) #

#jrinfer.seg_tests(solver, False, val, layer='score')
net_name = multilabel_accuracy.get_netname(testproto)
docker_hostname = socket.gethostname()
host_dirname = '/home/jeremy/caffenets/production'
Utils.ensure_dir(host_dirname)
baremetal_hostname = os.environ.get('HOST_HOSTNAME')
prefix = baremetal_hostname+'.'+net_name+docker_hostname
detailed_outputname = prefix + '.netoutput.txt'
detailed_jsonfile = detailed_outputname[:-4]+'.json'
loss_outputname = prefix + 'loss.txt'

copy2cmd = 'cp '+detailed_outputname + ' ' + host_dirname
copy3cmd = 'cp '+loss_outputname + ' ' + host_dirname
copy4cmd = 'cp '+detailed_jsonfile + ' ' + host_dirname
scp2cmd = 'scp '+detailed_outputname + ' root@104.155.22.95:/var/www/results/progress_plots/'
scp3cmd = 'scp '+loss_outputname+' root@104.155.22.95:/var/www/results/progress_plots/'
#scp4cmd = 'scp '+detailed_jsonfile + ' root@104.155.22.95:/var/www/results/progress_plots/'

Utils.ensure_file(loss_outputname)
Utils.ensure_file(detailed_outputname)

i = 0
losses = []
iters = []
steps_per_iter = 1
n_iter = 20
loss_avg = [0]*n_iter
tot_iters = 0

#instead of taking steps its also possible to do
#solver.solve()
#acc = single_label_accuracy.single_label_acc(weights,testproto,outlayer='fc2',n_tests=10,gpu=2,'ResNet-50_solver.prototxt')


for _ in range(100000):
    for i in range(n_iter):
        solver.step(steps_per_iter)
        loss = solver.net.blobs['loss'].data
        print('iter '+str(i*steps_per_iter)+' loss:'+str(loss))
        loss_avg[i] = loss
        losses.append(loss)
        tot_iters = tot_iters + steps_per_iter*n_iter
    averaged_loss=sum(loss_avg)/len(loss_avg)
    accuracy = solver.net.blobs['accuracy'].data
    print('avg loss over last {} steps is {}, acc:{}'.format(n_iter*steps_per_iter,averaged_loss,accuracy))
    #for test net:
#    solver.test_nets[0].forward()  # test net (there can be more than one)
    with open(loss_outputname,'a+') as f:
        f.write(str(int(time.time()))+'\t'+str(tot_iters)+'\t'+str(averaged_loss)+'\t'+str(accuracy)+'\n')
        f.close()
    jrinfer.seg_tests(solver, False, val, layer='conv_final',outfilename=detailed_outputname)
    #acc = single_label_accuracy.single_label_acc(weights,testproto,outlayer='fc2',n_tests=10,gpu=2,'ResNet-50_solver.prototxt')

    subprocess.call(copy2cmd,shell=True)
    subprocess.call(copy3cmd,shell=True)
#    subprocess.call(copy4cmd,shell=True)

    subprocess.call(scp2cmd,shell=True)
    subprocess.call(scp3cmd,shell=True)
#    subprocess.call(scp4cmd,shell=True)



