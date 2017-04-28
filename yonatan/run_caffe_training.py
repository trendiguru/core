import numpy as np
import os
import sys
import caffe

weights = '/data/yonatan/yonatan_files/prepared_caffemodels/ResNet-152-model.caffemodel'  #in brainia container jr2

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/data/yonatan/yonatan_files/trendi/yonatan/resnet_152_kaggle_planet/solver_152.prototxt')

# for finetune
solver.net.copy_from(weights)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)


#run net learning iterations
for _ in range(100000):
    solver.step(1)
    my_fc17 = solver.net.blobs['my_fc17'].data
    print('output of layer "my_fc17" {}'.format(my_fc17))
#    score.seg_tests(solver, False, val, layer='score')
#    jrinfer.seg_tests(solver, False, val, layer='score')
#    progress_plot.parse_solveoutput('net_output.txt')