__author__ = 'jeremy'
import scipy.optimize
import math

import numpy as np

#import fingerprint_core
import rate_fp
import NNSearch
from multiprocessing import Pool
import constants
import fingerprint_core

fingerprint_length=constants.fingerprint_length
n_docs = constants.max_items


def rate_wrapper(weights, k=0.5, image_sets=None, self_report=None):
    '''
    a wrapper to call self_rate_fingerprint without worrying about extra arguments, and also constrain weights to sum to 1;
    maybe this wrapper is not necessary given that u can call scipy.optimize.minimize(f,x0,args=(a,b,c)) to deal with fixed args a,b,c. Note
    if you only use one extra argument you have to use args=(a,) which is apparently a single tuple
    :param weights: the fingerprint weight vector
    :param k: distance power, k=1/2 gives euclidean distance
    :return:fingerprint rating as determined by self_rate_fingerprint
    '''
    for i in range(0,len(weights)):
        if weights[i] < 0:
            weights[i]=0
    sum = np.sum(weights)
    target = len(weights)
    weights=weights*float(target)/sum
    sum = np.sum(weights)
    print('constrained weights:'+str(weights))
    print('sum of weights after constraining:'+str(sum)+ ' target:'+str(target))
    rating, report = rate_fp.analyze_fingerprint(fingerprint_function=fingerprint_core.fp, weights=weights,
                                                 distance_function=NNSearch.distance_1_k, distance_power=k,
                                                 image_sets=image_sets, self_reporting=self_report)
    return rating

def optimize_weights(weights=np.ones(fingerprint_length),k=0.5):
    '''
    optimizes weights given everything else constant (fingerprint function, distance function, distance power k)
    :param k: distance power - k=0.5 is for euclidean distance
    :return:optimal weights vector (or at least best found so far)
    '''
    print('fp length:'+str(fingerprint_length))
    print('weights:'+str(weights))

#    x_min = scipy.optimize.minimize(f,initial_weights,args=(k),tol=0.1)
    f = rate_wrapper
    init=weights
# TO DO CONSTRAINED MINIMIZATION USE COBYLA  or SQLSQP - see docs - currentyly constraining 'by hand'
    # x_min = scipy.optimize.minimize(f,init,args=(k,),tol=0.1)   #in case u need only one optional argument this is how to do it
    self_report, image_sets = rate_fp.get_docs(n_docs)

    x_min = scipy.optimize.minimize(f, init, args=(k, image_sets, self_report), tol=0.1)
#    x_min = scipy.optimize.fmin(f,[2,3])
    print('output of optimize:'+str(x_min))
    print('xvals:'+str(x_min.x))
    print('f('+str(x_min.x)+')='+str(f(x_min.x)))



def parallel_optimize():
    p = Pool(5)
    f = optimize_weights
    print(p.map(f, [0.5,1,1.5]))

def test_function_vectorinput(x_vector):
    '''
    a test function whose optimal point should be at (6.66,7.66,8.66,...) for arbitrary length starting guess
    :return:minimum point
    '''

    print('input vector:'+str(x_vector))
    answer=[]
    #make n-d gaussian centered at 6.66,7.66 etc then flip upside-down
    for i in range(0,len(x_vector)):
        x_max = 6.66+i
        val = math.exp(-(x_vector[i]-x_max)**2)
        answer = np.append(answer,val)
    print('x values:'+str(x_vector)+' yvalues:'+str(answer))
    final_answer = -np.prod(answer)  #flip upside-down
    print('final answer:'+str(final_answer))
    return(final_answer)


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


def test_function(input_vector):
    answer = np.array([])
    x_vector = [x for x in input_vector]
    print('input vector:' + str(x_vector))
    for i in range(0, len(x_vector)):
        x_max = 6.66
        val = math.exp(-(x_vector[i] - x_max) ** 2)
        answer = np.append(answer, val)
    print('x values:' + str(x_vector) + ' yvalues:' + str(answer))
    final_answer = np.prod(answer)
    print('final answer:' + str(final_answer))
    return (final_answer)

def test2():
    f = test_function([1, 2, 3])
    print('f[10]=' + str(test_function([10])))
    x_min = scipy.optimize.minimize(test_function([10]), (3, 4, 5))
    print('output of optimize:'+str(x_min))
    print('xvals:'+str(x_min.x))
    # print('f(' + x_min.x + ')=' + str(f(x_min.x)))

if __name__ == '__main__':
   # parallel_optimize()
   optimize_weights()
