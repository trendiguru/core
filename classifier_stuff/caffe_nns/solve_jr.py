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
import numpy as np


from trendi import Utils
from trendi import constants
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi.classifier_stuff.caffe_nns import single_label_accuracy
from trendi.classifier_stuff.caffe_nns import multilabel_accuracy
from trendi.classifier_stuff.caffe_nns import progress_plot
from trendi.classifier_stuff.caffe_nns import caffe_utils

matplotlib.use('Agg') #allow plot generation on X-less systems
plt.ioff()
setproctitle.setproctitle(os.path.basename(os.getcwd()))



def dosolve(weights,solverproto,testproto,type='single_label',steps_per_iter=1,n_iter=200,n_loops=200,n_tests=1000,
          cat=None,classlabels=None,baremetal_hostname='brainiK80a',solverstate=None,label_layer='label',estimate_layer='my_fc2'):

    if classlabels is None:
        classlabels=['not_'+cat,cat]
    caffe.set_device(int(sys.argv[1]))
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solverproto)
    if weights is not None:
        solver.net.copy_from(weights)
    if solverstate is not None:
        solver.restore(solverstate)   #see https://github.com/BVLC/caffe/issues/3651
        #No need to use solver.net.copy_from(). .caffemodel contains the weights. .solverstate contains the momentum vector.
    #Both are needed to restart training. If you restart training without momentum, the loss will spike up and it will take ~50k i
    #terations to recover. At test time you only need .caffemodel.
    training_net = solver.net
    solver.test_nets[0].share_with(solver.net)  #share train weight updates with testnet
    test_net = solver.test_nets[0] # more than one testnet is supported

    #get netname, train_test train/test
    net_name = caffe_utils.get_netname(testproto)
    tt = caffe_utils.get_traintest_from_proto(solverproto)
    print('netname {} train/test {}'.format(net_name,tt))

    docker_hostname = socket.gethostname()

    datestamp = datetime.datetime.strftime(datetime.datetime.now(), 'time%H.%M_%d-%m-%Y')
    prefix = baremetal_hostname+'_'+net_name+'_'+docker_hostname+'_'+datestamp


    #detailed_jsonfile = detailed_outputname[:-4]+'.json'
    if weights:
        weights_base = os.path.basename(weights)
    else:
        weights_base = '_noweights_'
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



    #generate report filename, outdir to save everything (loss, html etc)
    if type == 'pixlevel':
        outname = os.path.join(outdir,outdir[2:]+'_netoutput.txt')  #TODO fix the shell script to not look for this, then it wont be needed
    if type == 'multilabel':
        outname = os.path.join(outdir,outdir[2:]+'_mlresults.html')
    if type == 'single_label':
        outdir = outdir + '_' + str(cat)
        outname = os.path.join(outdir,outdir[2:]+'_'+cat+'_slresults.txt')
    loss_outputname = os.path.join(outdir,outdir[2:]+'_loss.txt')
    print('outname:{}\n lossname {}\n outdir {}\n'.format(outname,loss_outputname,outdir))
    Utils.ensure_dir(outdir)
    time.sleep(0.1)
    Utils.ensure_file(loss_outputname)

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
    #cpoy solverproto to results dir
    if solverproto is not None:
        copycmd = 'cp '+solverproto + ' ' + outdir
        subprocess.call(copycmd,shell=True)
    #copy test proto to results dir
    if testproto is not None:
        copycmd = 'cp '+testproto + ' ' + outdir
        subprocess.call(copycmd,shell=True)
    #copy this file too
    copycmd = 'cp solve.py '  + outdir
    subprocess.call(copycmd,shell=True)


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
        multilabel_accuracy.open_html(weights, dir=outdir,solverproto=solverproto,caffemodel=weights,classlabels = constants.web_tool_categories_v2,name=outname)

    for _ in range(n_loops):
        for i in range(n_iter):
            solver.step(steps_per_iter)
    #        loss = solver.net.blobs['score'].data
            loss = solver.net.blobs['loss'].data
            loss_avg[i] = loss
            losses.append(loss)
            tot_iters = tot_iters + steps_per_iter
            if type == 'single_label':
                accuracy = solver.net.blobs['accuracy'].data
                accuracy_avg[i] = accuracy
                print('iter '+str(i*steps_per_iter)+' loss:'+str(loss)+' acc:'+str(accuracy))
            else:
                print('iter '+str(i*steps_per_iter)+' loss:'+str(loss))

        try:
            averaged_loss=sum(float(loss_avg))/len(loss_avg)
        except:
            print("something wierd with loss:"+str(loss_avg))
        s2 = '{}\t{}\n'.format(tot_iters,averaged_loss)
        #for test net:
    #    solver.test_nets[0].forward()  # test net (there can be more than one)
    #    progress_plot.lossplot(loss_outputname)  this hits tkinter problem
        if type == 'multilabel':
            precision,recall,accuracy,tp,tn,fp,fn = multilabel_accuracy.check_acc(test_net, num_samples=n_tests, threshold=0.5, gt_layer='labels',estimate_layer='prob')
            print('solve.py: p {} r {} a {} tp {} tn {} fp {} fn {}'.format(precision,recall,accuracy,tp,tn,fp,fn))
            n_occurences = [tp[i]+fn[i] for i in range(len(tp))]
            multilabel_accuracy.write_html(precision,recall,accuracy,n_occurences,threshold,weights,positives=True,dir=outdir,name=outname)
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss)

        elif type == 'pixlevel':
                    # number of tests for pixlevel
            s = 'avg loss over last {} steps is {}'.format(n_iter*steps_per_iter,averaged_loss)
            print(s)
            val = range(0,n_tests) #
            results_dict = jrinfer.seg_tests(solver,  val, output_layer=estimate_layer,gt_layer='label',outfilename=outname,save_dir=outdir,labels=labels)
            overall_acc = results_dict['overall_acc']
            mean_acc = results_dict['mean_acc']
            mean_ion = results_dict['mean_iou']
            fwavacc = results_dict['fwavacc']
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,overall_acc,mean_acc,mean_ion,fwavacc)

        elif type == 'single_label':
            averaged_acc = sum(float(accuracy_avg))/len(accuracy_avg)
            s = 'avg tr loss over last {} steps is {}, acc:{}'.format(n_iter*steps_per_iter,averaged_loss,averaged_acc)
            print(s)
            print accuracy_avg
            s2 = '{}\t{}\t{}\n'.format(tot_iters,averaged_loss,averaged_acc)

            acc = single_label_accuracy.single_label_acc(weights,testproto,net=test_net,label_layer='label',estimate_layer=estimate_layer,n_tests=n_tests,classlabels=classlabels,save_dir=outdir)
     #       test_net = solver.test_nets[0] # more than one testnet is supported
    #        testloss = test_net.blobs['loss'].data
            try:
                testloss =     test_net.blobs['loss'].data
            except:
                print('no testloss available')
                testloss=0
            with open(loss_outputname,'a+') as f:
                f.write('test\t'+str(int(time.time()))+'\t'+str(tot_iters)+'\t'+str(testloss)+'\t'+str(acc)+'\n')
                f.close()

        with open(loss_outputname,'a+') as f:
            f.write(str(int(time.time()))+'\t'+s2)
            f.close()
    #
    #   subprocess.call(copycmd,shell=True)
        subprocess.call(scpcmd,shell=True)



if __name__ == "__main__":
###############
#vars to change
###############
#ResNet-101-deploy.prototxt  ResNet-101-train_test.prototxt  ResNet-101_solver.prototxt  snapshot  solve.py
    solverstate = None
    base_dir = '/home/jeremy/caffenets/binary/resnet101_dress_try1/'
    weights =  '/home/jeremy/caffenets/binary/ResNet-101-model.caffemodel'
    solverproto = base_dir + 'ResNet-101_solver.prototxt'
    testproto = base_dir + 'ResNet-101-train_test.prototxt'
    type='single_label'
    #type='multilabel'
    #type='pixlevel'
    steps_per_iter = 1
    n_iter = 200
    cat = "dress"
    classlabels=['dress','not_dress']
    n_tests = 2000
    n_loops = 2000000
    baremetal_hostname = 'k80b'
    label_layer='label'
    estimate_layer='fc2'
    labels = constants.pixlevel_categories_v3
####################

    dosolve(weights,solverproto,testproto,type=type,steps_per_iter=steps_per_iter,n_iter=n_iter,n_loops=n_loops,n_tests=n_tests,
          cat=cat,classlabels=classlabels,baremetal_hostname=baremetal_hostname,label_layer=label_layer,estimate_layer=estimate_layer,
            labels=labels)
