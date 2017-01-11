"""
calcs concerning translation invaraiance of convolution+maxpool
"""
__author__ = 'jeremy'
import numpy as np

def calc_normal(n):
    n_wins = 0
    trials = 100000
    for i in range(trials):
        ylist = np.random.randn(n)
        if np.argmax(ylist) == 0:
#            print('winner!')
            n_wins+=1
    print('fraction {}'.format(float(n_wins)/trials))

if __name__ == "__main__":
    calc_normal(3)