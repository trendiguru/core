import argparse
from rq import Queue
from ...constants import db, redis_conn, redis_limit,features_per_category
from .renew_fp_worker import refresh_fp
from time import sleep
from ..general.db_utils import refresh_similar_results
from ..annoy_dir import fanni
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

    for col in ['amazon_US','amaze', 'ebay_US']:
        for gen in ['_Female','_Male']:
            col_name = col+gen
            print 'working on %s' % col_name
            collection = db[col_name]
            items = collection.find({},{'_id':1,'categories':1,'images.XLarge':1,'fingerprint':1}, no_cursor_timeout=True)

            for x,item in enumerate(items):
                renew_flag = False
                if divmod(x,50000)[1]==0:
                    print x
                try:
                    item_id = item['_id']
                    category = item['categories']
                    fp = item['fingerprint']
                    if fp is None:
                        renew_flag = True
                    elif type(fp) != dict:
                        fp = {'color': fp}
                    else:
                        pass
                    features_keys = features_per_category.keys()
                    if category in features_keys and not renew_flag:
                        wanted_keys = features_per_category[category]
                        fp_keys = fp.keys()
                        collar = False
                        if 'collar' in fp_keys:
                            if len(fp['collar'])==9:
                                collar = True
                        if any(key not in fp_keys for key in wanted_keys) or collar:
                            renew_flag = True

                except Exception as e:
                    print e
                    continue

                while q.count > redis_limit:
                    sleep(30)
                if renew_flag:
                    image_url = item['images']['XLarge']
                    q.enqueue(refresh_fp, args=(fp, col_name, item_id, category, image_url), timeout=1800)

            items.close()
            # fanni.plantForests4AllCategories(col_name)
        refresh_similar_results(col)