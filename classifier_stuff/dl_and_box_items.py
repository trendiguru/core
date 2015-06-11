__author__ = 'jeremy'

import numpy as np
import cv2


GREEN = [0, 255, 0]

__author__ = 'Nadav Paz'

import os

import pymongo

import Utils
import background_removal
from find_similar_mongo import get_all_subcategories


def dl_keyword_images(category_id, total=2000, keyword=None,
                      dir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/images',
                      show_visual_output=False):
    db = pymongo.MongoClient().mydb
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

        item_image = Utils.get_cv2_img_array(item['image']['sizes']['XLarge']['url'])
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

def get_items(categories, keywords):
    max_items_per_category = 2000
    for cat in cats:
        if keywords is not None:
            for keyword in keywords:
                print('getting cat ' + cat + ' w keyword ' + keyword)
                dl_keyword_images(cat, total=max_items_per_category, keyword=keyword, show_visual_output=True)
        else:
            print('getting cat ' + cat + ' no keyword ')
            dl_keyword_images(cat, total=max_items_per_category, show_visual_output=True)


def box_images(parent_dir='/home/ubuntu/Dev/'):
    for dir, subdir_list, file_list in os.walk(parent_dir):
        print('Found directory: %s' % dir)
        bbfilename = os.path.join(dir, 'bbs.txt')
        fp = open(bbfilename, 'a')
        for fname in file_list:
            print('\t%s' % fname)
            full_filename = os.path.join(dir, fname)
            # fp.write
            try:
                img_array = cv2.imread(fname)
                if img_array is None:
                    continue
                else:
                    # print('couldnt get locally (in not url branch)')
                    bb = get_bb(img_array)

            except:
                print("could not read locally, returning None")
                # logging.warning("could not read locally, returning None")
                continue  # input isn't a basestring nor a np.ndarray....so what is it?


def get_bb(img_array, show_visual_output=True):
    faces = background_removal.find_face(img_array)
    if len(faces):
        head_x0 = int(np.mean(face[0] for face in faces))
        head_y0 = int(np.mean(face[1] for face in faces))
        head_x1 = int(np.mean(face[2] for face in faces))
        head_y1 = int(np.mean(face[3] for face in faces))
        w = head_x1 - head_x0
        h = head_y1 - head_y0
        dress_w = w * 3
        dress_h = h * 6
        dress_x0 = head_x0 + w / 2 - dress_w / 2
        dress_y0 = head_y1
        dress_box = [dress_x0, dress_y0, dress_w, dress_h]
        if show_visual_output == True:
            cv2.rectangle(img_array, (dress_box[0], dress_box[1]),
                          (dress_box[0] + dress_box[2], dress_box[1] + dress_box[3]),
                          GREEN, thickness=1)
            cv2.imshow('im1', img_array)
            k = cv2.waitKey(200)

    else:
        return None


if __name__ == '__main__':
    print('starting')
    cats = ['cocktail-dresses', 'bridal-dresses', 'evening-dresses', 'day-dresses']
    keywords = ['mini', 'midi', 'maxi']
    get_items(['dresses'], keywords)
    get_items(cats, None)

    # check what all the subcats are
    # db = pymongo.MongoClient().mydb
    # subcategory_id_list = find_similar_mongo.get_all_subcategories(db.categories, cat)
    # print('sub cat id list for '+str(cat)+' is:'+str(subcategory_id_list))


    # Dresses
    #    Cocktail Dresses
    #    Bridal Dresses
    ##    Evening Dresses
    #   Day Dresses

    #get_