import redis
from rq import Connection, Queue, Worker
import sys
import matlab.engine
import redis
from redis import Redis

#redis_conn = Redis()
#q = Queue('pd', connection=redis_conn)
# Tell RQ what Redis connection to use



def my_enqueue(a,b):
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
    job = q.enqueue(my_function,a,b,eng)
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
