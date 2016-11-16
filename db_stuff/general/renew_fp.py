import argparse
from rq import Queue
from ...constants import db, redis_conn, redis_limit
from .renew_fp_worker import refresh_fp
from time import sleep
q = Queue('renew', connection=redis_conn)

def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ RENEW THE FP @@@')
    parser.add_argument('-c', '--collectiom',  dest="coollection_name",
                        help='enter full collection name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get user input
    user_input = get_user_input()
    collection_name = user_input.collection_name

    collection = db[collection_name]
    items = collection.find()

    for item in items:
        item_id = item['_id']
        category = item['categories']
        image_url = item['images']['XLarge']

        while q.count > redis_limit:
            sleep(30)

        q.enqueue(refresh_fp, args=(collection_name, item_id, category, image_url), timeout=1800)