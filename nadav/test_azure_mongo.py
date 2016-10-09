author = 'nadav'

import pymongo
import time
import numpy as np
from trendi import Utils, find_similar_mongo

client = pymongo.MongoClient(host="mongodb_mongodb_1")
db = client.mydb


def check_db_speed(url, products_collection, category, thresh):
    image = Utils.get_cv2_img_array(url)
    if image is None:
        print "Couldn't download image.."
        return
    mask = np.random.rand(image.shape[0], image.shape[1])
    mask = np.where(mask < thresh, 255, 0)
    start = time.time()
    find_similar_mongo.find_top_n_results(image=image, mask=mask, number_of_results=100, category_id=category,
                                          collection=products_collection, dibi=db)
    return time.time()-start