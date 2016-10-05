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
from trendi import constants
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy

setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'snapshot/train_0816__iter_25000.caffemodel'  #in brainia container jr2
solverproto = 'solver.prototxt'
testproto = 'train_test.prototxt'  #maybe take this out in  favor of train proto

weights = 'snapshot101_sgd/train_iter_70000.caffemodel'  #in brainia container jr2
solverproto = 'solver101_sgd.prototxt'
testproto = 'ResNet-101-test.prototxt'  #maybe take this out in  favor of train proto


caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()
#solver = caffe.SGDSolver('solver.prototxt')
#get_solver is more general, SGDSolver forces sgd even if something else is specified in prototxt
solver = caffe.get_solver(solverproto)
training_net = solver.net
solver.net.copy_from(weights)
solver.test_nets[0].share_with(solver.net)  #share train weight updates with testnet
test_net = solver.test_nets[0] # more than one testnet is supported

#solver.net.forward()  # train net  #doesnt do fwd and backwd passes apparently
# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#all_params = [k for k in solver.net.params.keys()]
#print('all params:')
#print all_params
#all_blobs = [k for k in solver.net.blobs.keys()]
#print('all blobs:')
#print all_blobs
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
prefix = baremetal_hostname+'_'+net_name+'_'+docker_hostname
#detailed_jsonfile = detailed_outputname[:-4]+'.json'
weights_base = os.path.basename(weights)
type='multilabel'
if net_name:
    outdir = type + '_' + prefix + '_' + weights_base.replace('.caffemodel','')
else:
    outdir = type + '_' + prefix + '_' +testproto+'_'+weights_base.replace('.caffemodel','')
outdir = outdir.replace('"','')  #remove quotes
outdir = outdir.replace(' ','')  #remove spaces
outdir = outdir.replace('\n','')  #remove newline
outdir = outdir.replace('\r','')  #remove return
outdir = './'+outdir
if type == 'pixlevel':
    outname = os.path.join(outdir,'netoutput.txt')  #TODO fix the shell script to not look for this, then it wont be needed
if type == 'multilabel':
    outname = os.path.join(outdir,'output.html')
if type == 'single_label':
    outname = os.path.join(outdir,outdir+'output.txt')
loss_outputname = os.path.join(outdir,'loss.txt')
print('outname:{} lossname {} outdir {}'.format(outname,loss_outputname,outdir))
Utils.ensure_dir(outdir)
time.sleep(0.1)
Utils.ensure_file(loss_outputname)

copy2cmd = 'cp '+outname + ' ' + host_dirname
copy3cmd = 'cp '+loss_outputname + ' ' + host_dirname
#copy4cmd = 'cp '+detailed_jsonfile + ' ' + host_dirname
scp2cmd = 'scp '+outname + ' root@104.155.22.95:/var/www/results/progress_plots/'
scp3cmd = 'scp '+loss_outputname+' root@104.155.22.95:/var/www/results/progress_plots/'
#scp4cmd = 'scp '+detailed_jsonfile + ' root@104.155.22.95:/var/www/results/progress_plots/'


i = 0
losses = []
iters = []
steps_per_iter = 1
n_iter = 20
loss_avg = [0]*n_iter
tot_iters = 0

#instead of taking steps its also possible to do
#solver.solve()
#acc = single_label_accuracy.single_label_acc(weights,testproto,net=test_net,label_layer='label',estimate_layer='loss',,n_tests=10,gpu=2,classlabels=['nond$

if type == 'multilabel':
    multilabel_accuracy.open_html(weights, dir=outdir,solverproto=solverproto,caffemodel=weights,classlabels = constants.web_tool_categories_v2)

for _ in range(1000000):
    for i in range(n_iter):
        solver.step(steps_per_iter)
        loss = solver.net.blobs['score'].data
#        loss = solver.net.blobs['loss'].data
        print('iter '+str(i*steps_per_iter)+' loss:'+str(loss))
        loss_avg[i] = loss
        losses.append(loss)
        tot_iters = tot_iters + steps_per_iter*n_iter
    averaged_loss=sum(loss_avg)/len(loss_avg)
    if type == 'single_label':
        accuracy = solver.net.blobs['accuracy'].data
        s = 'avg loss over last {} steps is {}, acc:{}'.format(n_iter*steps_per_iter,averaged_loss,accuracy)
        print(s)
        s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,accuracy)
    else:
        s = 'avg loss over last {} steps is {}'.format(n_iter*steps_per_iter,averaged_loss)
        print(s)
        s2 = '{}\t{}\n'.format(tot_iters,averaged_loss)
    #for test net:
#    solver.test_nets[0].forward()  # test net (there can be more than one)
    with open(loss_outputname,'a+') as f:
        f.write(str(int(time.time()))+'\t'+s2)
        f.close()
    if type == 'multilabel':
        precision,recall,accuracy,tp,tn,fp,fn = multilabel_accuracy.check_acc(test_net, num_samples=100, threshold=0.5, gt_layer='labels',estimate_layer='prob')
        print('solve.py: p {} r {} a {} tp {} tn {} fp {} fn {}'.format(precision,recall,accuracy,tp,tn,fp,fn))
        n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
        multilabel_accuracy.write_html(precision,recall,accuracy,n_occurences,t,weights,positives=True,dir=outdir)
    elif type == 'pixlevel':
        jrinfer.seg_tests(solver, False, val, layer='conv_final',outfilename=outname)
    elif type == 'single_label':
        acc = single_label_accuracy.single_label_acc(weights,testproto,outlayer='fc2',n_tests=10,gpu=2)

    subprocess.call(copy2cmd,shell=True)
    subprocess.call(copy3cmd,shell=True)
#    subprocess.call(copy4cmd,shell=True)

    subprocess.call(scp2cmd,shell=True)
    subprocess.call(scp3cmd,shell=True)
#    subprocess.call(scp4cmd,shell=True)



