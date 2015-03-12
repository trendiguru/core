__author__ = 'jeremy'
import numpy as np
import scipy.optimize
import math

def test_optimize():
    f = math.sin()
    x_min = scipy.optimize.minimize(f,3)
    print('output of optimize:'+str(x_min))
    print('xvals:'+str(x_min.x))
    print('f('+x_min.x+')='+f(x_min.x))

def f_multiple(x1,x2):
    print('xvals:'+str(x1)+','+str(x2))
    f = math.sin(x1*x2)
    print('f:'+str(f))
    return f

def opt_mult():
    f = test_function_vectorinput

#    x_min = scipy.optimize.minimize(f,init_val,args=(6))
    x_min = scipy.optimize.fmin(f,[2,3])
    print('output of optimize:'+str(x_min))

def test_function_vectorinput(x_arr):
    x_vector = x_arr
    print('input vector:'+str(x_vector))
    answer=[]
    for i in range(0,len(x_vector)):
        x_max = 6.66
        val = math.exp(-(x_vector[i]-x_max)**2)
        answer = np.append(answer,val)
    print('x values:'+str(x_vector)+' yvalues:'+str(answer))
    final_answer = -np.prod(answer)
    print('final answer:'+str(final_answer))
    return(final_answer)

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

def test2():
    f = test_function()
    print('f[10]='+str(test_function(3,4)))
    x_min = scipy.optimize.minimize(test_function(),(3,4,5))
    print('output of optimize:'+str(x_min))
    print('xvals:'+str(x_min.x))
    print('f('+x_min.x+')='+f(x_min.x))



opt_mult()