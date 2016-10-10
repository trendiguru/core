__author__ = 'jeremy'
import caffe
import time
import os
import sys
import setproctitle
import subprocess
import socket
import matplotlib
import matplotlib.pyplot as plt
from trendi import Utils
from trendi import constants
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import single_label_accuracy
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy
from trendi.classifier_stuff.caffe_nns import progress_plot

matplotlib.use('Agg') #allow plot generation on X-less systems
plt.ioff()
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'snapshot101_sgd/train_iter_70000.caffemodel'  #in brainia container jr2
solverproto = 'solver101_sgd.prototxt'
testproto = 'ResNet-101-test.prototxt'  #maybe take this out in  favor of train proto

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()
#solver = caffe.SGDSolver('solver.prototxt')
#get_solver is more general, SGDSolver forces sgd even if something else is specified in prototxt
solver = caffe.get_solver(solverproto)
training_net = solver.net
if weights is not None:
    solver.net.copy_from(weights)
solver.test_nets[0].share_with(solver.net)  #share train weight updates with testnet
test_net = solver.test_nets[0] # more than one testnet is supported

#solver.net.forward()  # train net  #doesnt do fwd and backwd passes apparently
# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

#all_params = [k for k in solver.net.params.keys()]
#all_blobs = [k for k in solver.net.blobs.keys()]


#jrinfer.seg_tests(solver, False, val, layer='score')
net_name = multilabel_accuracy.get_netname(testproto)
tt = multilabel_accuracy.get_traintest_from_proto(solverproto)

docker_hostname = socket.gethostname()
host_dirname = '/home/jeremy/caffenets/production'
Utils.ensure_dir(host_dirname)
baremetal_hostname = 'braini'

datestamp = datetime.datetime.strftime(datetime.datetime.now(), 'time%H.%M_%d-%m-%Y')
prefix = baremetal_hostname+'_'+net_name+'_'+docker_hostname+'_'+datestamp

#detailed_jsonfile = detailed_outputname[:-4]+'.json'
weights_base = os.path.basename(weights)
type='multilabel'
type='pixlevel'
type='single_label'
threshold = 0.5
if net_name:
    outdir = type + '_' + prefix + '_' + weights_base.replace('.caffemodel','')
else:
    outdir = type + '_' + prefix + '_' +testproto+'_'+weights_base.replace('.caffemodel','')
outdir = outdir.replace('"','')  #remove quotes
outdir = outdir.replace(' ','')  #remove spaces
outdir = outdir.replace('\n','')  #remove newline
outdir = outdir.replace('\r','')  #remove return
outdir = './'+outdir

if tt is not None:
    if len(tt) == 1:  #copy single traintest file to dir of info
        copycmd = 'cp '+tt[0] + ' ' + outdir
        subprocess.call(copycmd,shell=True)
    else:  #copy separate train and test files to dir of info
        copycmd = 'cp '+tt[0] + ' ' + outdir
        subprocess.call(copycmd,shell=True)
        copycmd = 'cp '+tt[1] + ' ' + outdir
        subprocess.call(copycmd,shell=True)


if type == 'pixlevel':
    outname = os.path.join(outdir,outdir[2:]+'_netoutput.txt')  #TODO fix the shell script to not look for this, then it wont be needed
if type == 'multilabel':
    outname = os.path.join(outdir,outdir[2:]+'_mlresults.html')
if type == 'single_label':
    outname = os.path.join(outdir,outdir[2:]+'_slresults.txt')
loss_outputname = os.path.join(outdir,outdir[2:]+'_loss.txt')
print('outname:{}\n lossname {}\n outdir {}\n'.format(outname,loss_outputname,outdir))
Utils.ensure_dir(outdir)
time.sleep(0.1)
Utils.ensure_file(loss_outputname)

copycmd = 'cp -r '+outdir + ' ' + host_dirname
#copy2cmd = 'cp '+outname + ' ' + host_dirname
#copy3cmd = 'cp '+loss_outputname + ' ' + host_dirname
#copy4cmd = 'cp '+detailed_jsonfile + ' ' + host_dirname
scpcmd = 'rsync -avz '+outdir + ' root@104.155.22.95:/var/www/results/'+type+'/'
#scp2cmd = 'scp '+outname + ' root@104.155.22.95:/var/www/results/progress_plots/'
#scp3cmd = 'scp '+loss_outputname+' root@104.155.22.95:/var/www/results/progress_plots/'
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

if type == 'multilabel':
    multilabel_accuracy.open_html(weights, dir=outdir,solverproto=solverproto,caffemodel=weights,classlabels = constants.web_tool_categories_v2,name=outname)

for _ in range(1000000):
    for i in range(n_iter):
        solver.step(steps_per_iter)
#        loss = solver.net.blobs['score'].data
        loss = solver.net.blobs['loss'].data
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
#    progress_plot.lossplot(loss_outputname)  this hits tkinter problem
    if type == 'multilabel':
        precision,recall,accuracy,tp,tn,fp,fn = multilabel_accuracy.check_acc(test_net, num_samples=100, threshold=0.5, gt_layer='labels',estimate_layer='prob')
        print('solve.py: p {} r {} a {} tp {} tn {} fp {} fn {}'.format(precision,recall,accuracy,tp,tn,fp,fn))
        n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
        multilabel_accuracy.write_html(precision,recall,accuracy,n_occurences,threshold,weights,positives=True,dir=outdir,name=outname)
    elif type == 'pixlevel':
                # number of tests for pixlevel
        val = range(0,200) #
        jrinfer.seg_tests(solver,  val, output_layer='mypixlevel_output',gt_layer='label',outfilename=outname,save_dir=outdir)

    elif type == 'single_label':
        acc = single_label_accuracy.single_label_acc(weights,testproto,net=test_net,label_layer='label',estimate_layer='fc2',n_tests=1000,classlabels=['not_item','item'],save_dir=outdir)
#
    subprocess.call(copycmd,shell=True)
    subprocess.call(scpcmd,shell=True)



