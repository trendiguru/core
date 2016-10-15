__author__ = 'jeremy'
from trendi.classifier_stuff.caffe_nns import solve

def vary_trainsize():
    #change number of trainingfiles
    solverstate = None
    weights = '../ResNet-101-model.caffemodel'  #in brainia container jr2
    solverproto = 'ResNet-101_solver.prototxt'
    testproto = 'ResNet-101-train_test.prototxt'  #maybe take this out in  favor of train proto
    type='single_label'
    #type='multilabel'
    #type='pixlevel'
    steps_per_iter = 1
    n_iter = 20
    cat = 'dress'
    classlabels=['not_'+cat,cat]
    n_tests = 1000
    n_outerloop = 200
    baremetal_hostname = 'M60'
    orig_trainfile = '/home/jeremy/image_dbs/tamara_berg_street_to_shop/dress_filipino_labels_balanced_train_250x250.txt'
    truncated_trainfile = './dress_filipino_labels_balanced_train_250x250_truncated.txt'
    for n in [500,1000,2000,5000,10000,20000,50000]:
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
        solve(weights,solverproto,testproto,type=type,steps_per_iter=steps_per_iter,n_iter=n_iter,n_outerloop=n_outerloop,n_tests=1000,
          cat='dress_n='+str(n),classlabels=classlabels,baremetal_hostname=baremetal_hostname)


#n_iter here 20->200 and in solverproto from 10->1 to increase precision of accuracy vals

if __name__ == "__main__":
    vary_trainsize()

#    solve('../ResNet-101-model.caffemodel',solverproto = 'ResNet-101_solver.prototxt',
#          testproto='ResNet-101-train_test.prototxt' ,type='single_label',cat='belt',
#          steps_per_iter=1,n_iter=20,n_loops=100,n_tests=1000,baremetal_hostname='brainik80',classlabels=None)
