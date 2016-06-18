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
import datetime
import time
import datetime

def parse_logfile(f):
  print('parsing logfile')
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

  past_beginning = False
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
#    if '] Iteration ' in line and 'loss = ' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      training_iterations.append(int(arr[0].strip(',')[4:]))
      training_loss.append(float(line.strip().split(' = ')[-1]))
      check_train = True


    if '] Iteration ' in line and 'Testing net' in line:
      arr = re.findall(r'ion \b\d+\b,', line)
      test_iterations.append(int(arr[0].strip(',')[4:]))
      check_test = True

    if '{' in line:
      past_beginning = True
    if not past_beginning and 'name' in line:
      net_name_arr = line.split('"')
      net_name = net_name_arr[-2]
      print('net name:'+net_name)

  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'train accuracy len: ', len(training_accuracy)
  print 'test iterations len: ', len(test_iterations)
  print 'test loss len: ', len(test_loss)
  print 'test accuracy len: ', len(test_accuracy)


  if len(test_iterations) != len(test_accuracy): #awaiting test...
    new_test_accuracy = []
    print 'mis-match'
    for i in range(0,len(test_accuracy))
      new_test_accuracy.append(test_accuracy[i])
    for i in range(len(test_accuracy),len(test_iterations))
      new_test_accuracy.append(-1)
    test_accuracy = new_test_accuracy

#    test_iterations = test_iterations[0:-1]

  f.close()
#  plt.plot(training_iterations, training_loss, '-', linewidth=2)
#  plt.plot(test_iterations, test_accuracy, '-', linewidth=2)
#  plt.show()

  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("iterations")
  host.set_ylabel("log loss")
  par1.set_ylabel("accuracy")

  p1, = host.plot(training_iterations, training_loss,'bo:', label="train logloss")
  p3, = host.plot(test_iterations, test_loss,'go:', label="test logloss")
  p2, = par1.plot(test_iterations, test_accuracy,'ro:', label="test acc.")
  if len(training_accuracy)>0:
    p4, = par1.plot(training_iterations, training_accuracy,'co:', label="train acc.")

#  par1.ylim((0,1))
#  host.legend(loc=2)

#top legend
#  host.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.1)

#right legend
  host.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)

  host.axis["left"].label.set_color(p1.get_color())
  par1.axis["right"].label.set_color(p2.get_color())

  dt=datetime.datetime.today()
  plt.title(net_name+' '+dt.isoformat())
  plt.suptitle(args.output_file)
  plt.draw()
  savename = args.output_file+'.jpg'
  plt.savefig(savename)
  plt.show()

def parse_solveoutput(f):
  print('parsing solve.py (jrinference) output')
  times = []
  training_iterations = []
  overall_accuracy = []
  acc_per_class = []
  mean_acc = []
  iou_per_class = []
  mean_iou = []
  fwavacc = []
  training_loss = []
  first_report = True
  extra_iters = 0

  for line in f:
#    print('checking line:'+line)
    if '>>>' in line:
      print line
      line = line.strip('>>>')
      thesplit = line.split()
      date=thesplit[0]
      yr,month,day =  date.split('-')
#      print('date {} yr {} month {} day {}'.format(date,yr,month,day))
      thetime=thesplit[1]
      hr,minutes,sec = thetime.split(':')
#      print('time {}  hr {} min {} sec {} '.format(thetime,hr,minutes,sec))
      dt = datetime.datetime(int(yr),int(month),int(day),int(hr),int(minutes),int(float(sec)))
      epochtime = time.mktime(dt.timetuple())
      if first_report:
        initial_time = epochtime
        first_report = False
      times.append(epochtime - initial_time)
      print('epoch:'+str(epochtime))
      iteration = int(thesplit[2].strip('Iteration:')) + extra_iters
      if len(training_iterations)>0 and iteration < training_iterations[-1]:
        print('iteration {} last in list {} extra iters before{}'.format(iteration,training_iterations[-1],extra_iters))
        extra_iters = training_iterations[-1]
        print('iteration {} last in list {} extra iters after {}'.format(iteration,training_iterations[-1],extra_iters))
        iteration =  extra_iters
      training_iterations.append(iteration)
      loss = thesplit[3].strip('loss:')
      training_loss.append(float(loss))
#      print('split :'+str(thesplit))
      print('time {} iter {} loss {}'.format(epochtime,iteration,loss))

    if 'acc per class' in line:
      vals = line.strip('acc per class:')
      vals = vals.strip('[')
      vals = vals.strip(']')
      vals = vals.split()
      print('vals:'+str(vals))
      acc_per_class.append([float(v) for v in vals])
      print('got acc per class:'+str(vals))
    if 'overall acc' in line:
      vals = line.strip('overall acc:')
      overall_accuracy.append(float(vals) )
      print('got overall acc :'+vals)
    if 'mean acc' in line:
      vals = line.strip('mean acc:')
      mean_acc.append(float(vals) )
      print('got mean acc :'+vals)

    if 'IU per class' in line:
      vals = line.strip('IU per class:')
      vals = vals.strip('[')
      vals = vals.strip(']')
      vals = vals.split()
      iou_per_class.append([float(v) for v in vals])
      print('got iou per class:'+str(vals))
    if 'mean IU' in line:
      vals = line.strip('mean IU:')
      mean_iou.append(float(vals) )
      print('got mean iou:'+vals)

    if 'fwavacc' in line:
      vals = line.strip('fwavacc:')
      fwavacc.append(float(vals) )
      print('got mean iou:'+vals)

  print 'train iterations len: ', len(training_iterations)
  print 'train loss len: ', len(training_loss)
  print 'OVERALL ACC len: ', len(overall_accuracy)
  print 'mean acc len: ', len(mean_acc)
  print 'fwvaccacc len: ', len(fwavacc)
  print 'mean iou len: ', len(mean_iou)
  print 'time len: ', len(times)

  elapsed_days = [float(t)/(3600.0*24) for t in times]

 # print times
 # print elapsed_days

  f.close()
#  plt.plot(training_iterations, training_loss, '-', linewidth=2)
#  plt.plot(test_iterations, test_accuracy, '-', linewidth=2)
#  plt.show()

  host = host_subplot(111)#, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  par1 = host.twinx()

  host.set_xlabel("iterations")
  host.set_ylabel("log loss, days elapsed")
  par1.set_ylabel("accuracy, iou")


  p1, = host.plot(training_iterations, training_loss,'bo:', label="train logloss")
  p3, = par1.plot(training_iterations, mean_acc,'go:', label="mean acc")
#  p4, = par1.plot(training_iterations, overall_accuracy,'ko:', label="overall_acc")
  p2, = par1.plot(training_iterations, fwavacc,'ro:', label="fwavacc")
  p5, = par1.plot(training_iterations, mean_iou,'co:', label="mean_iou")
#  p6, = host.plot(training_iterations, elapsed_days,'mo:', label="days_elapsed")
#  if len(training_accuracy)>0:
#    p4, = par1.plot(training_iterations, training_accuracy,'co:', label="train acc.")

#  par1.ylim((0,1))
#  host.legend(loc=2)

#top legend
#  host.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.1)

#right legend
  host.legend(bbox_to_anchor=(1.05, 1), loc=2,borderaxespad=0.)

  host.axis["left"].label.set_color(p1.get_color())
  par1.axis["right"].label.set_color(p2.get_color())

#  plt.title(net_name)
  dt=datetime.datetime.today()
  plt.title(dt.isoformat())
  plt.suptitle(args.output_file)
  plt.draw()
  savename = args.output_file+'.jpg'
  plt.savefig(savename)
  plt.show()


if __name__ == "__main__":
  print('starting')
  parser = argparse.ArgumentParser(description='makes a plot from Caffe output')
  parser.add_argument('output_file', help='file of captured stdout and stderr')
  parser.add_argument('--type', help='logfile or solve.py output',default='0')
  args = parser.parse_args()
  print('args:'+str(args))
  f = open(args.output_file, 'r')
  if args.type == '0':
    parse_logfile(f)
  elif args.type =='1':
    parse_solveoutput(f)