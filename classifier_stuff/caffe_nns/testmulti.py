__author__ = 'jeremy'
import multiprocessing
from trendi import simple_pool
import time

def squared(x):
    return x**2

def squared_cubed(x):
    return x**2,x**3

def noargs():
    return 3**2,3**3

def helpnoargs(X):
    r=noargs()
    return r

def test_multi(bsize):
#    bsize = 4
    pool = multiprocessing.Pool(20)
    ins = range(bsize)
#    outs = zip(*pool.map(squared, range(bsize)))
    outs = pool.map(squared_cubed, ins)
  #  outs = pool.map(helpnoargs, ins)
    print outs
    theout1=[]
    theout2=[]
    for o in outs:
        theout1.append(o[0])
        theout2.append(o[1])
    print theout1
    print theout2

class snorb():
    def msquared(self,x):
        return x**2

    def msquared_cubed(self,x):
        return x**2,x**3

    def mnoargs(self):
        time.sleep(1)
        return (3**2,3**3)

    def mhelpnoargs(self,x):
        r=self.mnoargs()
        time.sleep(1)

#        resq.put(r)
        #return r

    def test_multi(self,bsize):
    #    bsize = 4


        ins = range(5,bsize+5)

        print('ins:'+str(ins))
        outs = simple_pool.map2(self.mhelpnoargs, ins,ouput_ctype=(int))
#        outs = pool.map(self.msquared_cubed, ins)
        print('outs:'+str(outs))

        # result_queue = multiprocessing.Queue(10)
        # jobstodo = [self.mhelpnoargs(result_queue, fargs) for fargs in ins]
        # jobs = [multiprocessing.Process(j) for j in jobstodo]
        # for job in jobs: job.start()
        # for job in jobs: job.join()
        # results = [result_queue.get() for j in jobstodo]
        # print results





if __name__ == "__main__":
#    test_multi(6)
    start=time.time()
    s=snorb()
    s.test_multi(5)
    print('elapsed time'+str(time.time()-start))