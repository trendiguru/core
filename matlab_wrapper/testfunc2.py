#from matlab_queue import my_function

from rq import Connection, Queue, Worker

def my_function2(a=2,b=3):
    print('running function')
    return a+b

def enq(a,b):
    with Connection():
        q = Queue()

        async_result = q.enqueue(my_function2,a,b)


if __name__ == '__main__':
    # Tell rq what Redis connection to use
    with Connection():
        enq(10,100)
      #  Worker(q).work()
