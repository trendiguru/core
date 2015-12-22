__author__ = 'yonatan'

'''
same as all the other testers
'''

import time

from rq import Queue

from nate_worker import add_new_field
from trendi import constants

db = constants.db
redis = constants.redis_conn


def create_new_collection():
    collection = db.new_products

    category_stack = collection.find({"categories": "dress"})
    stack_length = category_stack.count()
    print(stack_length)
    db.nate_testing.remove()
    # Tell RQ what Redis connection to use

    q = Queue('nate_fp', connection=redis)  # no args implies the default queue)
    jobs = []
    for x, doc in enumerate(category_stack):
        if x > stack_length:
            break
        job = q.enqueue(add_new_field, doc, x)
        jobs.append(job)

    current = db.nate_testing.count()
    time.sleep(1)
    future = db.nate_testing.count()
    while current < future:
        time.sleep(60)
        current = future
        future = db.nate_testing.count()
        print future


if __name__ == "__main__":
    create_new_collection()
