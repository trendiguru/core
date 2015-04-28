__author__ = 'jeremy'

import scipy.optimize
import scipy.optimize
import math

import numpy as np

import fingerprint_core
import rate_fp
import NNSearch


fingerprint_length = 10



def rate_wrapper(weights,k):
    '''
    a wrapper to self_rate_fingerprint allowing external minimize function to call without dealing with extra args
    this can possibly be avoided using  scipy.optimize.minimize(f,x0,args=())
    :param weights: vector of weights to be optimized
    :param k: distance function power (1/2-> euclidean distance)
    :return:rating of current fingerprint
    '''
    print('initial weights:'+str(weights))
    for i in range(0,len(weights)):
        if weights[i] < 0:
            weights[i]=0
        i=i+1
    sum = np.sum(weights)
    print('sum before:'+str(sum))
    target = len(weights)
    weights=weights*float(target)/sum
    sum = np.sum(weights)
    print('corrected weights:'+str(weights))
    print('sum after:'+str(sum)+ ' trarget:'+str(target))
    rating = rate_fp.analyze_fingerprint(fingerprint_function=fingerprint_core.fp, weights=weights,
                                         distance_function=NNSearch.distance_1_k, distance_power=0.5)
    return rating

def optimize_weights(k):
    if k is None:
        k=0.5
  ##    initial_weights=np.ones(fingerprint_length)
#    initial_weights=np.random.random_integers(-2,2,fingerprint_length)
    f = math.sin

#    x_min = scipy.optimize.minimize(f,initial_weights,args=(k),tol=0.1)
    init=np.array(3)
    x_min = scipy.optimize.minimize(f,init,args=(k,),tol=0.1)
#    x_min = scipy.optimize.fmin(f,[2,3])
    print('output of optimize:'+str(x_min))

def test_function_vectorinput(x_arr):
    x_vector = x_arr
    print('input vector:'+str(x_vector))
    answer=[]
    for i in range(0,len(x_vector)):
        x_max = 6.66+i
        val = math.exp(-(x_vector[i]-x_max)**2)
        answer = np.append(answer,val)
    print('x values:'+str(x_vector)+' yvalues:'+str(answer))
    final_answer = -np.prod(answer)
    print('final answer:'+str(final_answer))
    return(final_answer)

def test_optimize():
    f = math.sin
    x_min = scipy.optimize.minimize(f,3)
    print('output of optimize:'+str(x_min))
    print('xvals:'+str(x_min.x))
    print('f('+x_min.x+')='+f(x_min.x))

def f_multiple(x1,x2):
    print('xvals:'+str(x1)+','+str(x2))
    f = math.sin(x1*x2)
    print('f:'+str(f))
    return f



def test_function_starinput(*input_vector):
    answer =np.array([])
    x_vector = [x for x in input_vector]
    print('input vector:'+str(x_vector))
    for i in range(0,len(x_vector)):
        x_max = 6.66
        val = math.exp(-(x_vector[i]-x_max)**2)
        answer = np.append(answer,val)
    print('x values:'+str(x_vector)+' yvalues:'+str(answer))
    final_answer = np.prod(answer)
    print('final answer:'+str(final_answer))
    return(final_answer)


def test_kwargs(positional1, **kwargs):
    print('positional:' + str(positional1))
    for arg in kwargs:
        print('arg:' + str(arg) + ' value:' + str(kwargs[arg]))
    passitalong('hello yourself', test1='again')

    passitalong('hello yourself', **kwargs)


def passitalong(positional, **kwargs):
    print('positional:' + str(positional))
    for arg in kwargs:
        print('arg:' + str(arg) + ' value:' + str(kwargs[arg]))

def test2():
    f = test_function_starinput()
    print('f[10]=' + str(test_function_starinput(3, 4)))
    x_min = scipy.optimize.minimize(test_function_starinput(), (3, 4, 5))
    print('output of optimize:'+str(x_min))
    print('xvals:'+str(x_min.x))
    # print('f(' + x_min.x + ')=' + str(f(x_min.x)))

def opt_mult():
    f = test_function_vectorinput
    x_min = scipy.optimize.minimize(f,[2,3])
#    x_min = scipy.optimize.fmin(f,[2,3])
    print('output of optimize:'+str(x_min))

#opt_mult()
if __name__ == "__main__":
    test_kwargs('hi', forbles='ee', snorbles=33)
    test_kwargs('hi')
    # optimize_weights(2)