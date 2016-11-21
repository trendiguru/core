import argparse
from rq import Queue
from ...constants import db, redis_conn, redis_limit
from .renew_fp_worker import refresh_fp
from time import sleep
from ..general.db_utils import refresh_similar_results
q = Queue('renew', connection=redis_conn)

def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ RENEW THE FP @@@')
    parser.add_argument('-c', '--collection',  dest="collection_name",
                        help='enter full collection name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get user input
    # user_input = get_user_input()
    # collection_name = user_input.collection_name

    for col in ['amazon_US','amazon_DE','amaze']:
        for gen in ['_Male','_Female']:
            col_name = col+gen
            print 'working on %s' % col_name
            collection = db[col_name]
            items = collection.find({},{'_id':1,'categories':1,'images':1,'fp':1}, no_cursor_timeout=True)

            for x,item in enumerate(items):
                if divmod(x,50000)[1]==0:
                    print x
                try:
                    item_id = item['_id']
                    category = item['categories']
                    if category not in ["dress", "top", "shirt", "t-shirt","sweater","sweatshirt","cardigan","blouse"]:
                        continue
                    fp = item['fp']
                    if type(fp)==dict:
                        if 'collar' in fp.keys():
                            continue

                    image_url = item['images']['XLarge']
                except:
                    continue

                while q.count > redis_limit:
                    sleep(30)

                q.enqueue(refresh_fp, args=(col_name, item_id, category, image_url), timeout=1800)
            items.close()
        refresh_similar_results(col)