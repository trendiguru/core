__author__ = 'jeremy'
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5,5,0.1)
sigmoid=np.divide(1,1+np.exp(-x))
ce_loss = (np.multiply(sigmoid,np.log(sigmoid))+np.multiply(1-sigmoid,np.log(1-sigmoid)))
plt.plot(x,sigmoid,label='$\sigma(x)=1/(1-exp(-x)) $')
plt.plot(x,ce_loss,label='$E(x)=\sigma(x)*log(\sigma(x) + \sigma(x)*log(\sigma(x)$')
plt.legend()
plt.show()