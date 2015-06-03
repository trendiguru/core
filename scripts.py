__author__ = 'Nadav Paz'

import urllib

import pymongo

import Utils
import background_removal
from find_similar_mongo import get_all_subcategories


def dl_keyword_images(keyword, category_id):
    db = pymongo.MongoClient().mydb
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    keyword_cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})
    path = '/home/ubuntu/Dev/' + keyword
    for dress in keyword_cursor:
        dress_image = Utils.get_cv2_img_array(dress['image']['XLarge']['url'])
        if background_removal.image_is_relevant(background_removal.standard_resize(dress_image, 400)[0]):
            urllib.urlretrieve(dress['image']['XLarge']['url'], path + '/' + dress['id'])
