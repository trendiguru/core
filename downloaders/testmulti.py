__author__ = 'jeremy'
from multiprocessing import Pool
import time

def f():
    return time.time(),7

def wrap(_):
    return f()

n=3
p=Pool(processes=n)
retval = p.map(wrap,range(3))
print('retval:'+str(retval))
#            p.close()
#            p.join()
       # p.map(convert_deepfashion_helper,(lines[i*n+j],fp2,labelfile,dir_to_catlist,visual_output,pardir ))


