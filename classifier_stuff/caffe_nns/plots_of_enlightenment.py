__author__ = 'jeremy'
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5,5,0.0001)
sigmoid=np.divide(1,1+np.exp(-x))
ce_loss = -(np.multiply(sigmoid,np.log(sigmoid))+np.multiply(1-sigmoid,np.log(1-sigmoid)))
logistic = -(np.multiply(x,np.log(x))+np.multiply(1-x,np.log(1-x)))
plt.plot(x,sigmoid,label='$\sigma(x)=1/(1-exp(-x)) $')
plt.plot(x,ce_loss,label='$-E(\sigma(x))=\sigma(x)*log(\sigma(x) + \sigma(1-x)*log(\sigma(1-x)$')
plt.plot(x,logistic,label='$-E(x)=x*log(x) + (1-x)*log(\sigma(1-x)$')
plt.legend(loc=2)
plt.tight_layout()
plt.xlim([-5,5])
plt.show()