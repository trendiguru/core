import numpy as np
import os
import sys
import caffe

weights = 'ResNet-50-model.caffemodel'  #in brainia container jr2

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/yonatan/trendi/yonatan/resnet_50_dress_sleeve_regression/solver50_sgd.prototxt')

# for finetune
solver.net.copy_from(weights)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)


#run net learning iterations
for _ in range(1000):
    solver.step(5000)
    my_fc8 = solver.net.blobs['my_fc8'].data
    print('output of layer "my_fc8" {}'.format(my_fc8))
#    score.seg_tests(solver, False, val, layer='score')
#    jrinfer.seg_tests(solver, False, val, layer='score')
#    progress_plot.parse_solveoutput('net_output.txt')