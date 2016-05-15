__author__ = 'kaggle_guy'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import sys
import argparse
import re
from pylab import figure, show, legend, ylabel

from mpl_toolkits.axes_grid1 import host_subplot


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='makes a plot from Caffe output')
  parser.add_argument('output_file', help='file of captured stdout and stderr')
  args = parser.parse_args()

  f = open(args.output_file, 'r')

  training_iterations = []
  training_accuracy = []
  training_loss = []

  test_iterations = []
  test_accuracy = []
  test_loss = []

  check_test = False
  check_test2 = False
  check_train = False
  check_train2 = False

  for line in f:
#    print('checking line:'+line)
    if check_test and 'Test net output' in line and 'accuracy' in line:
      print('checking line for test output 0: '+line)
      test_accuracy.append(float(line.strip().split(' = ')[-1]))
      print('got test accuracy :'+str(float(line.strip().split(' = ')[-1])))
      check_test = False
      check_test2 = True
    elif check_test2:
      print('checking line for test output1:'+line)
      if 'Test net output' in line and 'loss' in line:
        #print line
        test_loss.append(float(line.strip().split(' ')[-2]))
        print('got test loss:'+str(line.strip().split(' ')[-2]))
        check_test2 = False
      else:
        test_loss.append(0)
        check_test2 = False

    if check_train  and 'Train net output' in line and 'accuracy' in line:
      print('checking line for train output acc:'+line)
      training_accuracy.append(float(line.strip().split(' = ')[-1]))
      print('got train acc :'+str(float(line.strip().split(' = ')[-1])))
      check_train = False
      check_train2 = True


    if '] Iteration ' in line and 'loss = ' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      training_iterations.append(int(arr[0].strip(',')[4:]))
      training_loss.append(float(line.strip().split(' = ')[-1]))
      check_train = True


    if '] Iteration ' in line and 'Testing net' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      test_iterations.append(int(arr[0].strip(',')[4:]))
      check_test = True

  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'train accuracy len: ', len(training_accuracy)
  print 'test iterations len: ', len(test_iterations)
  print 'test loss len: ', len(test_loss)
  print 'test accuracy len: ', len(test_accuracy)

  if len(test_iterations) != len(test_accuracy): #awaiting test...
    print 'mis-match'
    print len(test_iterations[0:-1])
    test_iterations = test_iterations[0:-1]

  f.close()
#  plt.plot(training_iterations, training_loss, '-', linewidth=2)
#  plt.plot(test_iterations, test_accuracy, '-', linewidth=2)
#  plt.show()

  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("iterations")
  host.set_ylabel("log loss")
  par1.set_ylabel("validation accuracy")

  p1, = host.plot(training_iterations, training_loss, label="training log loss")
  p3, = host.plot(test_iterations, test_loss, label="valdation log loss")
  p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
  if len(training_accuracy)>0:
    p4, = par1.plot(training_iterations, training_accuracy, label="training accuracy")

  host.legend(loc=3)

  host.axis["left"].label.set_color(p1.get_color())
  par1.axis["right"].label.set_color(p2.get_color())

  plt.draw()
  savename = args.output_file+'.jpg'
  plt.savefig(savename)
  plt.show()
