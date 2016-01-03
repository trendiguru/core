__author__ = 'yonatan'

'''
same as all the other testers
'''

from rq import Queue

from nate_worker import add_new_field
from ... import constants

db = constants.db
redis = constants.redis_conn


def create_new_collection():
    collection = db.new_products

    # category_stack = collection.find({"categories": "dress"})
    category_stack = db.nate_testing.find()
    stack_length = category_stack.count()
    print(stack_length)
    # db.nate_testing.remove()
    # Tell RQ what Redis connection to use

    q = Queue('nate_fp', connection=redis)  # no args implies the default queue)
    jobs = []
    for x, doc in enumerate(category_stack):
        # if x < 5000:
        #     continue
        # job = q.enqueue_call(func=add_new_field, args=(doc, x), ttl=1000, result_ttl=1000, timeout=1000)
        # jobs.append(job)
        add_new_field(doc, x)
    # current = db.nate_testing.count()
    # time.sleep(1)
    # future = db.nate_testing.count()
    # while current < future:
    #     time.sleep(60)
    #     current = future
    #     future = db.nate_testing.count()
    #     print future
    print ("finished")

if __name__ == "__main__":
    create_new_collection()
