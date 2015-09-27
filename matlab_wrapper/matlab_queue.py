#from rq import Queue
from redis import Redis
# -*- coding: utf-8 -*-
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)
from rq import Connection, Queue, Worker
import sys
#redis_conn = Redis()
#q = Queue('pd', connection=redis_conn)

# Tell RQ what Redis connection to use

def my_enqueue(a,b):
    print sys.path
    redis_conn = Redis()
    q = Queue(connection=redis_conn)
    job = q.enqueue(my_function,a,b)
    return job.result

def my_function(a=2,b=3):
    print('running function')
    return a+b

if __name__ == '__main__':
    # Tell rq what Redis connection to use
    with Connection():
        q = Queue()
        Worker(q).work()
        async_result = q.enqueue(my_function,10,100)
