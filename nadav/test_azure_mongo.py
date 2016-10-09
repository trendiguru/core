author = 'nadav'

import pymongo
import time
import numpy as np
from trendi import Utils, find_similar_mongo, fingerprint_core, NNSearch

client = pymongo.MongoClient(host="mongodb_mongodb_1")
db = client.mydb
number_of_matches = 100


def check_db_speed(url, products_collection, category, annoy_list):
    image = Utils.get_cv2_img_array(url)
    if image is None:
        print "Couldn't download image.."
        return
    mask = np.random.rand(image.shape[0], image.shape[1])
    mask = np.where(mask < 0.1, 255, 0).astype(np.uint8)
    start = time.time()
    fp = fingerprint_core.dict_fp(image, mask, category)
    # find_similar_mongo.find_top_n_results(image=image, mask=mask, number_of_results=100, category_id=category,
    #                                       collection=products_collection, dibi=db)
    # entries = db[products_collection].find({'categories': category},
    #                                        {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1})
    entries = db[products_collection].find({"AnnoyIndex": {"$in": annoy_list}, 'categories': category},
                                           {"id": 1, "fingerprint": 1, "images.XLarge": 1, "clickUrl": 1},).hint([('AnnoyIndex', 1)])
    farthest_nearest = 1
    nearest_n = []
    # tt = 0
    i = 0
    # for i, entry in enumerate(entries):
    for entry in entries:
        i += 1
        # t1 = time()
        # tt += t1-t2
        ent = entry['fingerprint']
        d = NNSearch.distance(category, fp, ent, products_collection)
        if not d:
            # t2 = time()
            # tdif = t2 - t1
            # tt += tdif
            continue
        if i < number_of_matches:
            nearest_n.append((entry, d))
            farthest_nearest = 1
        else:
            if i == number_of_matches:
                # sort by distance
                nearest_n.sort(key=lambda tup: tup[1])
                # last item in the list (index -1, go python!)
                farthest_nearest = nearest_n[-1][1]

            # Loop through remaining entries, if one of them is better, insert it in the correct location and remove last item
            if d < farthest_nearest:
                insert_at = number_of_matches - 2
                while d < nearest_n[insert_at][1]:
                    insert_at -= 1
                    if insert_at == -1:
                        break
                nearest_n.insert(insert_at + 1, (entry, d))
                nearest_n.pop()
                farthest_nearest = nearest_n[-1][1]
        # t2 = time()
        # tdif = t2-t1
        # tt+=tdif
    # print tt

    # print "sorting entries took {0} secs".format(time()-start)
    # t3 = time()
    [result[0].pop('fingerprint') for result in nearest_n]
    [result[0].pop('_id') for result in nearest_n]
    nearest_n = [result[0] for result in nearest_n]
    # t4 = time()
    # print t4-t3
    # return nearest_n
    return time.time()-start