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

setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'snapshot/train_0816__iter_25000.caffemodel'  #in brainia container jr2

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
#solver.net.forward()  # train net  #doesnt do fwd and backwd passes apparently
# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
all_layers = [k for k in solver.net.params.keys()]
print('all layers:')
print all_layers
#surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('../data/segvalid11.txt', dtype=str)
val = range(0,200) #

#jrinfer.seg_tests(solver, False, val, layer='score')
net_name = 'MYNET'
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
accuracy_avg = [0]*n_iter
tot_iters = 0
with open(loss_outputname,'a+') as f:
    f.write('time \t tot_iters \t averaged_loss \t accuracy\n')
    f.close()
for _ in range(100000):
    for i in range(n_iter):
        solver.step(steps_per_iter)
        loss = solver.net.blobs['loss'].data
        loss_avg[i] = loss
        accuracy = solver.net.blobs['accuracy'].data
        accuracy_avg[i] = accuracy
        losses.append(loss)
        tot_iters = tot_iters + steps_per_iter*n_iter
        print('iter '+str(tot_iters)+' loss:'+str(loss)+' acc:'+str(accuracy))
    averaged_loss=sum(loss_avg)/len(loss_avg)
    averaged_acc=sum(accuracy_avg)/len(accuracy_avg)
    print('avg loss over last {} steps is {}, acc {}'.format(n_iter*steps_per_iter,averaged_loss,accuracy_avg))
    with open(loss_outputname,'a+') as f:
        f.write(str(int(time.time()))+'\t'+str(tot_iters)+'\t'+str(averaged_loss)+'\t'+str(accuracy_avg)+'\n')
        f.close()
#    jrinfer.seg_tests(solver, False, val, layer='conv_final',outfilename=detailed_outputname)
    subprocess.call(copy2cmd,shell=True)
    subprocess.call(copy3cmd,shell=True)
#    subprocess.call(copy4cmd,shell=True)

    subprocess.call(scp2cmd,shell=True)
    subprocess.call(scp3cmd,shell=True)
#    subprocess.call(scp4cmd,shell=True)



