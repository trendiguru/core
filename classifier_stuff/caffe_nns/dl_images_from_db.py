__author__ = 'jeremy'
import os
import logging
import time

import cv2
from rq import Queue
from operator import itemgetter
import json
from trendi import constants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from trendi.constants import db
from trendi.constants import redis_conn
import trendi.Utils as Utils
from trendi.find_similar_mongo import get_all_subcategories

current_directory_name = os.getcwd()
my_path = os.path.dirname(os.path.abspath(__file__))

# download_images_q = Queue('download_images', connection=redis_conn)  # no args implies the default queue
logging.basicConfig(level=logging.WARNING)
MAX_IMAGES = 10000

#dress
#pants jeans
#shirt top
    # tees-and-tshirts
#sweaters
#sweatshirts athletic-tops athletic-jackets
#outerwear jackets  coats
#blazers
#suit
#shorts
#longsleeve-tops', 'shortsleeve-tops', 'sleeveless-to'
#skirts', 'mini-skirts', 'mid-length-skirts', 'long-skirts


#list of outermost cats and synonyms, for searching in our db
#(CANNOT PUT MULTIPLE PHRASES IN $text)


def get_db_fields(collection='products'):
    db = constants.db
    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.products.find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get collection"}
    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        for topic,value in doc.iteritems():
            try:
                print(str(topic))
                print(str(value))
            except UnicodeEncodeError:
                print('unicode encode error')
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}


def step_thru_db(collection='products'):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    db = constants.db
    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.products.find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get colelction"}
    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        if 'categories' in doc:
            try:
                print('cats:' + str(doc['categories']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['categories']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        if 'description' in doc:
            try:
                print('desc:' + str(doc['description']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['description']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}

def find_products_by_description_and_category(search_string, category_id):
    logging.info('****** Starting to find {0} in category {1} *****'.format(search_string,category_id))

    query = {"$and": [{"$text": {"$search": search_string}},
                      {"categories":
                           {"$elemMatch":
                                {"id": {"$in": get_all_subcategories(db.categories, category_id)}
                                 }
                            }
                       }]
             }
    fields = {"categories": 1, "id": 1, "description": 1}
    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in cat {category} with string {search_string}".format(count=cursor.count(),
                                                                    category=category_id,
                                                                    search_string=search_string))
    return cursor

def find_products_by_category(category_id):
    logging.info('****** Starting to find category {} *****'.format(category_id))

    query = {"categories":
                           {"$elemMatch":
                                {"id": {"$in": get_all_subcategories(db.categories, category_id)}
                                 }
                            }
             }
    fields = {"categories": 1, "id": 1, "description": 1}
    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in cat {category} ".format(count=cursor.count(),
                                                                    category=category_id))
    return cursor


def simple_find_products_by_category(category_id):
    logging.info('****** Starting to find category {} *****'.format(category_id))

#{"username" : {$regex : ".*son.*"}}
    reg = "*"+category_id+"*"
    query = {"categories":
                           {"$regex":reg}
                   }
    fields = {"categories": 1, "id": 1, "description": 1}
#    cursor = db.products.find(query, fields).batch_size(10)
    cursor = db.products.find(query).batch_size(10)
    logging.info("Found {count} products in cat {category} ".format(count=cursor.count(),
                                                                    category=category_id))
    return cursor


def enqueue_for_download(q, iterable, feature_name, category_id, max_images=MAX_IMAGES):
    job_results = []
    for prod in iterable:
        res = q.enqueue(download_image, prod, feature_name, category_id, max_images)
        job_results.append(res.result)
    return job_results

def download_image(prod, feature_name, category_id, max_images):
    downloaded_images = 0
    directory = os.path.join(category_id, feature_name)
    try:
        downloaded_images = len([name for name in os.listdir(directory) if os.path.isfile(name)])
    except:
        pass
    if downloaded_images < max_images:
            xlarge_url = prod['image']['sizes']['XLarge']['url']

            img_arr = Utils.get_cv2_img_array(xlarge_url)
            if img_arr is None:
                logging.warning("Could not download image at url: {0}".format(xlarge_url))
                return

            relevance = background_removal.image_is_relevant(img_arr)
            if relevance.is_relevant:
                logging.info("Image is relevant...")

                filename = "{0}_{1}.jpg".format(feature_name, prod["id"])
                filepath = os.path.join(directory, filename)
                Utils.ensure_dir(directory)
                logging.info("Attempting to save to {0}...".format(filepath))
                success = cv2.imwrite(filepath, img_arr)
                if not success:
                    logging.info("!!!!!COULD NOT SAVE IMAGE!!!!!")
                    return 0
                # downloaded_images += 1
                logging.info("Saved... Downloaded approx. {0} images in this category/feature combination"
                             .format(downloaded_images))
                return 1
            else:
                # TODO: Count number of irrelevant images (for statistics)
                return 0

def run(category_id, search_string_dict=None, async=True):
    logging.info('Starting...')
    download_images_q = Queue('download_images', connection=redis_conn, async=async)
    search_string_dict = search_string_dict or descriptions_dict

    job_results_dict = dict.fromkeys(descriptions_dict)

    for name, search_string_list in search_string_dict.iteritems():
        for search_string in search_string_list:
            cursor = find_products_by_description(search_string, category_id, name)
            job_results_dict[name] = enqueue_for_download(download_images_q, cursor, name, category_id, MAX_IMAGES)

    while True:
        time.sleep(10)
        for name, jrs in job_results_dict.iteritems():
            logging.info(
                "{0}: Downloaded {1} images...".format(name,
                                                       sum((job.result for job in jrs if job and job.result))))

def print_logging_info(msg):
    print msg


# hackety hack
logging.info = print_logging_info

if __name__ == '__main__':
    descriptions_dict = {'dress': ["dress"],
                     'skirt': ["skirt"],
                     'pants': ["pants", "jeans"],
                     'shirt': ["shirt", "top"],
                     'outerwear': ["outerwear", "jacket","coat"],
                     'suit': ["suit","blazer"],
                     'shorts': ["shorts"]}
    for key,val in descriptions_dict.iteritems():
        for cat in val:
            cursor = simple_find_products_by_category(cat)
            n=cursor.count()
            print('cat:{0} subcat:{1} n:{2}'.format(key,cat,n))
