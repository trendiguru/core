__author__ = 'Nadav Paz'

import urllib
import os
import pymongo
import Utils
import background_removal
from find_similar_mongo import get_all_subcategories
from .constants import db


def dl_keyword_images(category_id, total=3000000, keyword=None):
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    if keyword is None:
        path = '/home/ubuntu/Dev/' + category_id
        cursor = db.products.find(query)
    else:
        path = '/home/ubuntu/Dev/' + keyword
        cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    for item in cursor:
        if i > total:
            break
        i += 1
        item_image = Utils.get_cv2_img_array(item['image']['sizes']['XLarge']['url'])
        if background_removal.image_is_relevant(background_removal.standard_resize(item_image, 400)[0]):
            urllib.urlretrieve(item['image']['sizes']['XLarge']['url'], path + '/' + str(item['id']) + '.jpg')


def clean_duplicates(collection, field):
    collection = db[collection]
    before = collection.count()
    sorted = collection.find().sort(field, pymongo.ASCENDING)
    print('starting, total {0} docs'.format(before))
    current_url = ""
    i = deleted = 0
    for doc in sorted:
        i += 1
        if i % 1000 == 0:
            print("deleted {0} docs after running on {1}".format(deleted, i))
        if doc['image_urls'][0] != current_url:
            current_url = doc['image_urls'][0]
            deletion = collection.delete_many({'$and': [{'image_urls': doc['image_urls'][0]}, {'_id': {'$ne': doc['_id']}}]}).deleted_count
            if deletion:
                deleted += deletion
                print("found duplicates to {0}".format(doc['image_urls'][0]))
    print("total {0} docs were deleted".format(deleted))


if __name__ == '__main__':
    print('starting')
    clean_duplicates('images', 'image_urls')
