__author__ = 'jeremy'
import multiprocessing

def squared(x):
    return x**2

def squared_cubed(x):
    return x**2,x**3

def noargs():
    return 3**2,4**3

def helpnoargs(X):
    r=noargs()
    return r

def test_multi(bsize):
#    bsize = 4
    pool = multiprocessing.Pool(20)
    ins = range(bsize)
#    outs = zip(*pool.map(squared, range(bsize)))
    outs = pool.map(squared_cubed, ins)
    outs = pool.map(helpnoargs, [None,None,None])
    print outs
    theout1=[]
    theout2=[]
    for o in outs:
        theout1.append(o[0])
        theout2.append(o[1])
    print theout1
    print theout2

if __name__ == "__main__":
    test_multi(6)