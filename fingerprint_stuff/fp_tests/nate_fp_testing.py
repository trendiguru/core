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
    category_stack = db.nate_testing.find({"sp_update": {"$exists": 0}}).batch_size(15000)
    stack_length = category_stack.count()
    print(stack_length)
    # db.nate_testing.remove()
    # Tell RQ what Redis connection to use

    q = Queue('nate_fp', connection=redis)  # no args implies the default queue)
    jobs = []
    for doc in category_stack:
        # if x < 5000:
        #     continue
        db.nate_testing.find_one_and_update({'id': doc['id']},
                                            {"$set": {"sp_update": "Queued"}})
        job = q.enqueue(add_new_field, doc)
        jobs.append(job)
        # try:
        #     add_new_field(doc, x)
        # except:
        #     print("error in add_new_field - file not updated")
        #     db.nate_testing.delete_one({"id": doc["id"]})
    # current = db.nate_testing.count()
    # time.sleep(1)
    # future = db.nate_testing.count()
    # while current < future:
    #     time.sleep(60)
    #     current = future
    #     future = db.nate_testing.count()
    #     print future
    print ("finished creating Q")

if __name__ == "__main__":
    create_new_collection()
