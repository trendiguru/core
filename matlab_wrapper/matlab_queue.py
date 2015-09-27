import redis
from rq import Connection, Queue, Worker
import sys
import matlab.engine
import redis
from redis import Redis

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
    print('engine names:'+str(engines))
    if engines[0] is not None:
        print('connecting to engine:'+str(engines[0]))
        eng = matlab.engine.connect_matlab(engines[0])
        test = eng.factorial(6)
        print('matlab thinks 6!='+str(test))

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
    job = q.enqueue(my_function,a,b,eng)
    print('after enqueuing function')
    print('result:'+str(job.result))
    return job.result

def my_function(a=2,b=3,eng=None):
    if eng is None:
        print('got no engine so starting on my own')
        eng = matlab.engine.start_matlab('-nodesktop')
    print('running function')
    return eng.factorial(a+b)

if __name__ == '__main__':
    # Tell rq what Redis connection to use
    with Connection():
        q = Queue()
        Worker(q).work()
        async_result = q.enqueue(my_function,10,100)
