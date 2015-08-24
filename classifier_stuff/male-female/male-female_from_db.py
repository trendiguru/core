import os
import logging
import time

import cv2
import pymongo
from rq import Queue
from redis import Redis

from ... import Utils
from ... import background_removal
from ...find_similar_mongo import get_all_subcategories












# Tell RQ what Redis connection to use
redis_conn = Redis()
# download_images_q = Queue('download_images', connection=redis_conn)  # no args implies the default queue
logging.basicConfig(level=logging.WARNING)
db = pymongo.MongoClient().mydb

MAX_IMAGES = 10000

# Leftovers:
descriptions = ['male', 'mens',
                'female', 'womens']

# LESSONS: CANNOT PUT MULTIPLE PHRASES IN $text
# v-neck is a superset of v-neckline
descriptions_dict = {'womens': ["women", "female", "womens"],
                     'mens': ["men", "male", "mens"]}


def find_products_by_description(search_string, category_id, feature_name=None):
    if category_id is None:
        logging.warning('no category given')
        return
    if db.categories is None:
        logging.warning('no db categories')
        return
    logging.info('****** Starting to find {0} *****'.format(feature_name))
    try:
        query = {"$and": [{"$text": {"$search": search_string}},
                          {"categories":
                               {"$elemMatch":
                                    {"id": {"$in": get_all_subcategories(db.categories, category_id)}
                                     }
                                }
                           }]
                 }
    except:
        logging.warning('exception in get_all_subcategories')
        return
    if query is None:
        logging.warning('query is none')
        return

    fields = {"categories": 1, "image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1,
              "id": 1, "description": 1, "feature_bbs": 1}

    downloaded_images = 0

    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in {category} with {feature}".format(count=cursor.count(),
                                                                              category=category_id,
                                                                              feature=feature_name))
    return cursor


def enqueue_for_download(q, iterable, feature_name, category_id, max_images=MAX_IMAGES):
    job_results = []
    if iterable is None:
        logging.warning('iterable in enq for dl is None')
        return
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
    category_id = 'mens-clothing'
    run(category_id)
