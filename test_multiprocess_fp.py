__author__ = 'liorsabag'

import multiprocessing as mp
import time
import random
from Utils import ThreadSafeCounter
from .constants import db
# import signal

CONTINUE = mp.Value("b", True)
Q = mp.Queue(25)
TOTAL_TASKS = mp.Value("i", 0)
NUM_PROCESSES = mp.Value("i", 0)
CURRENT = ThreadSafeCounter()


def fake_fp(product, p_time=.5):
    global CURRENT
    CURRENT.increment()
    print "{0} working on product {1}. {2} of {3}...\n".format(str(mp.current_process().name), product["id"], CURRENT.value, TOTAL_TASKS.value)
    # print str(current_process().pid) + " working on: " + str(product["id"]) + "\n"
    start_time = time.time()
    while time.time() < start_time + p_time:
        start_time * start_time
        # print "done: " + str(product["id"])


def do_work_on_q(some_func, q):
    print "Planning on doing some work..."
    try:
        while CONTINUE.value:
            popped_item = q.get()
            if popped_item is None:
                print "Process {0} finished".format(str(mp.current_process().pid))
                return

            some_func(popped_item)
    except BaseException as e:
        print "Exception in do_work: {0}".format(e)
    return "{0} returned".format(str(mp.current_process().pid))


def feed_q(q):
    global TOTAL_TASKS, NUM_PROCESSES
    mini_skirt_cursor = db.products.find({"categories": {"$elemMatch": {"id": "mini-skirts"}}},
                                         {"id": 1, "images": 1}).batch_size(12000)[0:100]

    TOTAL_TASKS.value = mini_skirt_cursor.count()
    print "Total tasks: {0}".format(str(TOTAL_TASKS.value))

    for doc in mini_skirt_cursor:
        q.put(doc)

    for p in range(0, NUM_PROCESSES.value):
        q.put(None)

    print "Done putting all docs in Q"
    q.close()


def worker_done(result):
    print result

if __name__ == "__main__":

    NUM_PROCESSES.value = 4  # int(mp.cpu_count() * .75)

    print "Main process ID: {0}".format(mp.current_process().pid)

    feeder = mp.Process(target=feed_q, name="feeder", args=[Q])

    worker_list = [mp.Process(target=do_work_on_q, name="Worker {0}".format(i), args=(fake_fp, Q))
                   for i in range(0, NUM_PROCESSES.value)]

    # worker_pool = mp.Pool(processes=NUM_PROCESSES.value, initializer=init_do_work, initargs=[Q])

    print "Starting Feeder"
    feeder.start()

    print "Starting workers"
    for p in worker_list:
        p.start()
    # async_results = [worker_pool.apply_async(do_work, args=(fake_fp,), callback=worker_done)
    #                  for i in range(NUM_PROCESSES.value)]

    print "Waiting for workers to return"
    for p in worker_list:
        p.join()

    feeder.join()

    print "All Done"