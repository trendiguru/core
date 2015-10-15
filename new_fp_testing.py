__author__ = 'yonatan'

'''
these are the steps required for testing a new fingerprint:
1. create a new branch
2. update the fingerprint_core.py file
3. update the constants.py file
4. create a new collection for the testing
    - use the create_new_collection function
    - run on the server from the main directory and not from the branch
    'python -m [branchname].new_fp_testing'
    - each item in the new collection will have 2 fingerprints - the old('fingerprint) and the new('new_fp')
    - this should take about 40 minutes
5. use the testing_demo to display the results
    'http://extremeli.trendi.guru/demo/new_fp_testing_demo/demo-day.html'
'''

import time

from rq import Queue
from redis import Redis
from .constants import db

from fp_testing_workers import add_new_fp


def create_new_collection():

    collection = db.products

    category_stack = collection.find({"categories": {"$elemMatch": {"id": "day-dresses"}}})
    stack_length = category_stack.count()
    db.fp_testing.remove()
    # Tell RQ what Redis connection to use
    redis_conn = Redis()

    q = Queue('fingerprint', connection=redis_conn)  # no args implies the default queue)
    jobs = []
    for x, doc in enumerate(category_stack):
        job = q.enqueue(add_new_fp, doc, x)
        jobs.append(job)

    current = db.fp_testing.count()
    while current < (stack_length - 1000):
        time.sleep(10)
        current = db.fp_testing.count()
        print current


if __name__ == "__main__":
    create_new_collection()
