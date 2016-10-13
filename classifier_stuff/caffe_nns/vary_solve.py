__author__ = 'jeremy'

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
import datetime
from trendi import Utils
from trendi import constants
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import single_label_accuracy
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy
from trendi.classifier_stuff.caffe_nns import progress_plot

matplotlib.use('Agg') #allow plot generation on X-less systems
plt.ioff()
setproctitle.setproctitle(os.path.basename(os.getcwd()))


def solve(weightsfile='../ResNet-101-model.caffemodel',solverproto = 'ResNet-101_solver.prototxt',
          testproto='ResNet-101-train_test.prototxt' ,type='single_label',cat='belt',
          steps_per_iter=1,n_iter=20,n_loops=100,n_tests=1000,baremetal_hostname='brainik80',classlabels=None):
    if classlabels is None:
        classlabels = ['not_'+cat,cat]
    caffe.set_device(int(sys.argv[1]))
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solverproto)
    training_net = solver.net
    if weightsfile is not None:
        solver.net.copy_from(weightsfile)
    solver.test_nets[0].share_with(solver.net)  #share train weight updates with testnet
    test_net = solver.test_nets[0] # more than one testnet is supported

    net_name = multilabel_accuracy.get_netname(testproto)
    tt = multilabel_accuracy.get_traintest_from_proto(solverproto)
    print('netname {} train/test {}'.format(net_name,tt))

    docker_hostname = socket.gethostname()

    datestamp = datetime.datetime.strftime(datetime.datetime.now(), 'time%H.%M_%d-%m-%Y')
    prefix = baremetal_hostname+'_'+net_name+'_'+docker_hostname+'_'+datestamp

    #detailed_jsonfile = detailed_outputname[:-4]+'.json'
    weights_base = os.path.basename(weightsfile)
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

    #copy training and test files to outdir
    if tt is not None:
        if len(tt) == 1:  #copy single traintest file to dir of info
            copycmd = 'cp '+tt[0] + ' ' + outdir
            subprocess.call(copycmd,shell=True)
        else:  #copy separate train and test files to dir of info
            copycmd = 'cp '+tt[0] + ' ' + outdir
            subprocess.call(copycmd,shell=True)
            copycmd = 'cp '+tt[1] + ' ' + outdir
            subprocess.call(copycmd,shell=True)

    #generate report filename
    if type == 'pixlevel':
        outname = os.path.join(outdir,outdir[2:]+'_netoutput.txt')  #TODO fix the shell script to not look for this, then it wont be needed
    if type == 'multilabel':
        outname = os.path.join(outdir,outdir[2:]+'_mlresults.html')
    if type == 'single_label':
        outdir = outdir + '_' + cat
        outname = os.path.join(outdir,outdir[2:]+'_'+cat+'_slresults.txt')
    loss_outputname = os.path.join(outdir,outdir[2:]+'_loss.txt')
    print('outname:{}\n lossname {}\n outdir {}\n'.format(outname,loss_outputname,outdir))
    Utils.ensure_dir(outdir)
    time.sleep(0.1)
    Utils.ensure_file(loss_outputname)

    #copycmd = 'cp -r '+outdir + ' ' + host_dirname
    scpcmd = 'rsync -avz '+outdir + ' root@104.155.22.95:/var/www/results/'+type+'/'

    i = 0
    losses = []
    iters = []
    loss_avg = [0]*n_iter
    accuracy_avg = [0]*n_iter
    tot_iters = 0

    #instead of taking steps its also possible to do
    #solver.solve()

    if type == 'multilabel':
        multilabel_accuracy.open_html(weightsfile, dir=outdir,solverproto=solverproto,caffemodel=weightsfile,classlabels = constants.web_tool_categories_v2,name=outname)

    for _ in range(n_loops):
        for i in range(n_iter):
            solver.step(steps_per_iter)
    #        loss = solver.net.blobs['score'].data
            loss = solver.net.blobs['loss'].data
            print('iter '+str(i*steps_per_iter)+' loss:'+str(loss))
            loss_avg[i] = loss
            losses.append(loss)
            tot_iters = tot_iters + steps_per_iter*n_iter
            if type == 'single_label':
                accuracy_avg[i] = solver.net.blobs['accuracy'].data
        averaged_loss=sum(loss_avg)/len(loss_avg)
        if type == 'single_label':
            averaged_acc = sum(accuracy_avg)/len(accuracy_avg)
            s = 'avg loss over last {} steps is {}, acc:{}'.format(n_iter*steps_per_iter,averaged_loss,averaged_acc)
            print(s)
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,averaged_acc)
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
            precision,recall,accuracy,tp,tn,fp,fn = multilabel_accuracy.check_acc(test_net, num_samples=n_tests, threshold=0.5, gt_layer='labels',estimate_layer='prob')
            print('solve.py: p {} r {} a {} tp {} tn {} fp {} fn {}'.format(precision,recall,accuracy,tp,tn,fp,fn))
            n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
            multilabel_accuracy.write_html(precision,recall,accuracy,n_occurences,threshold,weights,positives=True,dir=outdir,name=outname)
        elif type == 'pixlevel':
                    # number of tests for pixlevel
            val = range(0,n_tests) #
            jrinfer.seg_tests(solver,  val, output_layer='mypixlevel_output',gt_layer='label',outfilename=outname,save_dir=outdir)

        elif type == 'single_label':
            acc = single_label_accuracy.single_label_acc(weights,testproto,net=test_net,label_layer='label',estimate_layer='fc2',n_tests=n_tests,classlabels=classlabels,save_dir=outdir)
     #       test_net = solver.test_nets[0] # more than one testnet is supported
    #        testloss = test_net.blobs['loss'].data
            testloss = 0
            with open(loss_outputname,'a+') as f:
                f.write('test\t'+str(int(time.time()))+'\t'+str(tot_iters)+'\t'+str(testloss)+'\t'+str(acc)+'\n')
                f.close()
    #
    #   subprocess.call(copycmd,shell=True)
        subprocess.call(scpcmd,shell=True)



def vary_trainsize():
    #change number of trainingfiles
    orig_trainfile = '/home/jeremy/image_dbs/tamara_berg_street_to_shop/dress_filipino_labels_balanced_train_250x250.txt'
    truncated_trainfile = '/home/jeremy/image_dbs/tamara_berg_street_to_shop/dress_filipino_labels_balanced_train_250x250_truncated.txt'
    for n in [100,200,500,1000,2000,5000,10000,20000,50000]:
        with open(orig_trainfile,'r') as fp:
            lines = fp.readlines()
            first_n = lines[0:n]
            fp.close()
#        Utils.ensure_file(truncated_trainfile)
        with open(truncated_trainfile,'w') as fp2:
            for line in first_n:
                fp2.write(line)
            fp2.close
        print('n {}'.format(n))
     #   raw_input()
        solve('../../ResNet-101-model.caffemodel',solverproto = 'ResNet-101_solver.prototxt',
          testproto='ResNet-101-train_test.prototxt' ,type='single_label',cat='dress',
          steps_per_iter=1,n_iter=1,n_loops=1,n_tests=10,baremetal_hostname='brainik80',classlabels=None)


if __name__ == "__main__":
    vary_trainsize()

#    solve('../ResNet-101-model.caffemodel',solverproto = 'ResNet-101_solver.prototxt',
#          testproto='ResNet-101-train_test.prototxt' ,type='single_label',cat='belt',
#          steps_per_iter=1,n_iter=20,n_loops=100,n_tests=1000,baremetal_hostname='brainik80',classlabels=None)
