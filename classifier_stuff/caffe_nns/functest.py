__author__ = 'jeremy'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def expfunc(x,asymptote,timeconst,x0):
    eps = 10**-5
    y = asymptote * (1-np.exp(-(x-x0)/(timeconst)))
    print('as {} tm {} x0 {}'.format(asymptote,timeconst,x0))

    return y
#    return a * np.exp(-b * x) + c
#a * np.exp(-b * x) + c

def fit_points_exp(xlist,ylist):
    p0 = {'asymptote':0.8,'timeconst':2000,'y0':0.7}
    p0 = [0.8,2000,0]
    popt,pcov = curve_fit(expfunc, xlist, ylist,p0=p0, sigma=None, absolute_sigma=False)
    print('popt {} pcov {}'.format(popt,pcov))
    n_timeconstants = xlist[-1]/popt[1]
    return(popt,n_timeconstants)

def test_fit():
#    x=np.linspace(0,10000,100)
#    y=expfunc(x,0.8,2000,100)
#    y = y + 0.2*np.random.rand(len(y))*y

    with open('/home/jeremy/projects/core/classifier_stuff/caffe_nns/loss3.txt','r') as fp:
        r = fp.readlines()
        x = [int(l.split()[1]) for l in r]
        y = [float(l.split()[-1]) for l in r ]
        x = np.array(x)
        y=np.array(y)
        print x,y
        popt,nt = fit_points_exp(x,y)
        y_est = expfunc(x,popt[0],popt[1],popt[2])
        n_timeconstants = x[-1]/popt[1]
        print('ntimesconsts {} nt {}'.format(n_timeconstants,nt))
        plt.plot(x,y,x,y_est)
        plt.show()
        print y_est

