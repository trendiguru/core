import redis
from rq import Connection, Queue, Worker, SimpleWorker
import sys
import matlab.engine
import redis
from redis import Redis
import time
#redis_conn = Redis()
#q = Queue('pd', connection=redis_conn)
# Tell RQ what Redis connection to use



def zero_engines():
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    r.set('n_matlab_engines',0)


def my_enqueue(a,b):
    #names = matlab.engine.find_matlabmatlab.engine.shareEngine('Engine_1');
#    matlab.engine.start_matlab	Start MATLAB Engine for Python
#    x=matlab.engine.connect_matlab()
#    print('x:'+str(x))
#matlab.engine.shareEngine	Convert running MATLAB session to shared session
#matlab.engine.engineName	Return name of shared MATLAB session
#matlab.engine.isEngineShared
    eng = None
    engines=matlab.engine.find_matlab()	#Find shared MATLAB sessions to connect to MATLAB Engine for Python
#    print('engine names:'+str(engines))

    if(0):
        print('attempting to queue')
        r = redis.StrictRedis(host='localhost', port=6379, db=0)
        try:
            n_running_engines = r.get('n_matlab_engines')
            if n_running_engines is None:
                print('got no # engines')
                r.set('n_matlab_engines',0)
                n_running_engines = 0

            else:
                print('got '+str(n_running_engines)+' engines')

        except:
            r.set('n_matlab_engines',0)
            n_running_engines = 0

        if(n_running_engines>0):
            eng = r.get('matlab_engine')
            print('got engine '+str(eng))

    redis_conn = Redis()
    q = Queue(connection=redis_conn)
#    job = q.enqueue('self.matlab_engine.factorial',a,b)
    print('enqueuing function')
    tryagain = True
    n=0
    print('len:'+str(len(engines)))
    while(tryagain is True and n<len(engines) and 0):
        try:
            print('connecting to engine:'+str(engines[n]))
            eng = matlab.engine.connect_matlab(engines[n])
            test = eng.factorial(6)
            print('matlab thinks 6!='+str(test))
            job = q.enqueue(my_function,a,b)
            tryagain=False
#        except MatlanxecutionError:
#            print('ML execution error after enqueuing function')
#            tryagain = False
#        except EngineError:
        except:
            print('caught exception')
            n=n+1
            eng = matlab.engine.connect_matlab(engines[n])
            tryagain=True

#    queue = Queue(connection=Redis())
#    queue.enqueue(my_long_running_job)
#    worker = Worker
#    sworker = SimpleWorker
#    sworker.work
    job = q.enqueue(my_function,a,b)
    print('job result:'+str(job.result))
    while job.result is None:
        time.sleep(1)
        print('job result:'+str(job.result))

    return job.result

def my_function(a,b, eng):
    print('starting queue function')

#    if eng is None:
#        print('got no engine so  on my own')
#        eng = matlab.engine.start_matlab('-nodesktop')
    print('running  with eng '+str(eng))
#    return('hi')
    return eng.factorial(a+b)

if __name__ == '__main__':
    # Tell rq what Redis connection to use
    with Connection():
        q = Queue()
        Worker(q).work()
        async_result = q.enqueue(my_function,10,100)
