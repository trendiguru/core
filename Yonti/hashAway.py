'''
parallel hashing of a collection
'''
import sys
import argparse
from .. import constants
from rq import Queue
from ..Yonti.hash_worker import get_hash
q = Queue('hash_q', connection=constants.redis_conn)
db = constants.db


def progress_bar(val, end_val, bar_length=50):
    percent = float(val) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rScraping: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def hashCollection(collection_name):
    collection = db[collection_name]
    items = collection.find({'img_hash': {'$exists': 0}}, {'images.XLarge': 1})
    total_count = items.count()
    print (total_count)
    for x, item in enumerate(items):
        print (x)
        q.enqueue(get_hash, collection_name=collection_name, item_count = x, item_id = item["_id"],
                  item_url=item["images"]["XLarge"])
        progress_bar(x, total_count)

    print ('all items sent to hash')
    return


def getUserInput():
    parser = argparse.ArgumentParser(description='hash master')
    parser.add_argument("-c", dest="collection_name", help="The collection to be hashed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    user_input = getUserInput()
    print user_input
    hashCollection(user_input.collection_name)
    sys.exit(0)