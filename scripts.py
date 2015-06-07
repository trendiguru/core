__author__ = 'Nadav Paz'

import urllib
import os

import pymongo

import Utils
import background_removal
from find_similar_mongo import get_all_subcategories


def dl_keyword_images(category_id, keyword=None, total=3000000):
    db = pymongo.MongoClient().mydb
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    if keyword is None:
        path = '/home/ubuntu/Dev/' + category_id
        keyword_cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})
    else:
        path = '/home/ubuntu/Dev/' + keyword
        keyword_cursor = db.products.find(query)
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    for item in keyword_cursor:
        if i > total:
            break
        i += 1
        dress_image = Utils.get_cv2_img_array(item['image']['sizes']['XLarge']['url'])
        if background_removal.image_is_relevant(background_removal.standard_resize(dress_image, 400)[0]):
            urllib.urlretrieve(item['image']['sizes']['XLarge']['url'], path + '/' + str(dress['id']) + '.jpg')
