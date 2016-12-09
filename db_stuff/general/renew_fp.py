import argparse
from rq import Queue
from ...constants import db, redis_conn, redis_limit,features_per_category
from .renew_fp_worker import refresh_fp
from time import sleep
from ..general.db_utils import refresh_similar_results
from ..annoy_dir import fanni
from ... import Utils, background_removal
import logging
q = Queue('renew', connection=redis_conn)


def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ RENEW THE FP @@@')
    parser.add_argument('-c', '--collection',  dest="collection_name",
                        help='enter full collection name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get user input
    user_input = get_user_input()
    col = user_input.collection_name


    for gen in ['_Female','_Male']:
        col_name = col+gen
        print 'working on %s' % col_name
        collection = db[col_name]
        items = collection.find({},{'_id':1,'categories':1,'images.XLarge':1,'fingerprint':1}, no_cursor_timeout=True)

        for x,item in enumerate(items):
            renew_flag = False
            if divmod(x, 10000)[1] == 0:
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
                    if 'collar' in fp_keys:
                        if fp['collar'] is None or len(fp['collar'])==9:
                            renew_flag = True
                    for k in fp_keys:
                        if fp[k] is None:
                            renew_flag = True
                    if any(key not in fp_keys for key in wanted_keys):
                        renew_flag = True

            except Exception as e:
                print e
                continue

            while q.count > redis_limit:
                sleep(30)
            if renew_flag:
                image_url = item['images']['XLarge']
                image = Utils.get_cv2_img_array(image_url)
                if not Utils.is_valid_image(image):
                    logging.warning("image is None. url: {url}".format(url=image_url))
                    collection.delete_one({'_id': item_id})
                    continue

                small_image, resize_ratio = background_removal.standard_resize(image, 400)

                if not Utils.is_valid_image(small_image):
                    logging.warning("small_image is Bad. {img}".format(img=small_image))
                    collection.delete_one({'_id': item_id})
                    continue
                q.enqueue(refresh_fp, args=(fp, col_name, item_id, category, image, small_image), timeout=1800)

        items.close()
        # fanni.plantForests4AllCategories(col_name)
        refresh_similar_results(col)
