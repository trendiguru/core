__author__ = 'Nadav Paz'

import urllib
import os

import pymongo
import cv2

import Utils
import background_removal
from find_similar_mongo import get_all_subcategories


def dl_keyword_images(category_id, total=3000000, keyword=None, dir='home/ubuntu/Dev', show_visual_output=False):
    db = pymongo.MongoClient().mydb
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    if keyword is None:  # make dir ~/home/ubuntu/Dev/category if theres no keyword
        path = os.path.join(dir, category_id)
        cursor = db.products.find(query)
    else:  # make dir ~/home/ubuntu/Dev/category/keyword   if there is a keyword
        path = os.path.join(dir, category_id)
        path = os.path.join(path, keyword)
        cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})
    print(str(cursor.count()) + ' results found for ' + category_id)
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    for item in cursor:
        if i > total:
            break
        i += 1
        item_image = Utils.get_cv2_img_array(item['image']['sizes']['XLarge']['url'])
        if show_visual_output == True:
            cv2.imshow('im1', item_image)
            k = cv2.waitKey(200)
        # if background_removal.image_is_relevant(background_removal.standard_resize(item_image, 400)[0]):
        if background_removal.image_is_relevant(item_image):
            urllib.urlretrieve(item['image']['sizes']['XLarge']['url'], path + '/' + str(item['id']) + '.jpg')
            print('downloaded ' + str(item['id']) + '.jpg')
        else:
            print('image not relevant')
