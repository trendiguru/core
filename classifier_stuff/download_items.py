from __future__ import with_statement
__author__ = 'dr. groovemaster'
import cv2

GREEN = [0, 255, 0]
RED = [0, 0, 255]
BLUE = [255, 0, 0]

__author__ = 'dr. juice-man'
# theirs
import os

# ours
import Utils
import background_removal
from find_similar_mongo import get_all_subcategories
from ..constants import db

def dl_keyword_images(category_id, total=2000, keyword=None,
                      # dir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/images',
                      # dir='/home/ubuntu/Dev/trendi_guru_modules/images',
                      dir='images',
                      show_visual_output=False):
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    if keyword is None:
        path = os.path.join(dir, category_id)
        cursor = db.products.find(query)
    else:
        path = os.path.join(dir, category_id)
        path = os.path.join(path, keyword)
        cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})
    print('path:' + path)
    if not os.path.exists(path):
        print('creating dir')
        os.makedirs(path)
    i = 0
    for item in cursor:
        if i > total:
            break
        i += 1
        url = item['image']['sizes']['XLarge']['url']
        print('url:' + url)
        item_image = Utils.get_cv2_img_array(url)
        if item_image is None:
            return None
        if show_visual_output == True:
            cv2.imshow('im1', item_image)
            k = cv2.waitKey(200)

        if background_removal.image_is_relevant(background_removal.standard_resize(item_image, 400)[0]):
            name = os.path.join(path, str(item['id']) + '.jpg')
            try:
                print('writing ' + name)
                cv2.imwrite(name, item_image)
            except:
                print('couldnt write file:' + name)


def get_items(categories, keywords=None, dir=None):
    max_items_per_category = 10000
    for cat in categories:
        if keywords is not None:
            for keyword in keywords:
                print('getting category ' + cat + ' w keyword ' + keyword)
                dl_keyword_images(cat, total=max_items_per_category, keyword=keyword, show_visual_output=False, dir=dir)
        else:
            print('getting category ' + cat + ' no keyword ')
            dl_keyword_images(cat, total=max_items_per_category, show_visual_output=False)


if __name__ == '__main__':
    print('starting')
    cats = ['cocktail-dresses', 'bridal-dresses', 'evening-dresses', 'day-dresses']
    cats = ['mens-shirts', 'womens-tops']
    cats = ['skirts']
    keywords = ['mini', 'midi', 'maxi']

    # from scipy import misc
    # l = misc.lena()
    # import matplotlib.pyplot as plt
    # plt.imshow(l)
    # plt.show()

    dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/classifier_stuff/images/dresses'
    dir = 'images/skirts'
    get_items(cats, keywords=None, dir=dir)
    # get_items(cats, None)

    # check what all the subcats are
    # db = pymongo.MongoClient().mydb
    # subcategory_id_list = find_similar_mongo.get_all_subcategories(db.categories, cat)
    # print('sub cat id list for '+str(cat)+' is:'+str(subcategory_id_list))


    # Dresses
    # Cocktail Dresses
    # Bridal Dresses
    ##    Evening Dresses
    # Day Dresses

    # get_