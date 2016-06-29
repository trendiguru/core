__author__ = 'jeremy'
import numpy as np
import matplotlib.pyplot as plt

##NNOTE CE LOSS IS L=tlog(p)+(1-t)*log(1-p) where t=target and p=prob!!! not plogp!!
x=np.arange(-5,5,0.0001)
plt.figure(1)
#plt.subplot(211)
logistic = -(np.multiply(x,np.log(x))+np.multiply(1-x,np.log(1-x)))
sigmoid=np.divide(1,1+np.exp(-x))
plt.plot(x,sigmoid,label='$\sigma(x)=1/(1-exp(-x)) $')
plt.plot(x,logistic,label='$-E(x)=x*log(x) + (1-x)*log((1-x)$')
plt.legend(loc=2)
plt.tight_layout()
plt.xlim([-5,5])

#plt.subplot(212)
#x=np.arange(-1,1,0.0001)
sigmoid=np.divide(1,1+np.exp(-x))
ce_loss = -(np.multiply(sigmoid,np.log(sigmoid))+np.multiply(1-sigmoid,np.log(1-sigmoid)))
plt.figure(2)
plt.plot(sigmoid,ce_loss,label='$-E(\sigma(x))=\sigma(x)*log(\sigma(x) + \sigma(1-x)*log(\sigma(1-x)$')
#plt.axis([0, 1, 0, 1])
plt.legend(loc=2)
plt.tight_layout()
plt.xlim([0,1])
plt.show()