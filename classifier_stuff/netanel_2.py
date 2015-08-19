import os
import logging

import cv2
import pymongo
from rq import Queue
from redis import Redis

from .. import Utils
from .. import background_removal
from ..find_similar_mongo import get_all_subcategories



# Tell RQ what Redis connection to use
redis_conn = Redis()
download_images_q = Queue('download_images', connection=redis_conn)  # no args implies the default queue
# save_relevant_q = Queue('save_relevant', connection=redis_conn)  # no args implies the default queue


# logging.basicConfig(level=logging.INFO)

db = pymongo.MongoClient().mydb

MAX_IMAGES = 10000

def find_and_download_images(feature_name, search_string, category_id, max_images):

    logging.info('****** Starting to find {0} *****'.format(feature_name))

    query = {"$and": [{"$text": {"$search": search_string}},
                      {"categories":
                           {"$elemMatch":
                                {"id": {"$in": get_all_subcategories(db.categories, category_id)}
                                 }
                            }
                       }]
             }
    fields = {"categories": 1, "image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1,
              "id": 1, "description": 1, "feature_bbs": 1}

    downloaded_images = 0

    cursor = db.products.find(query, fields).batch_size(10)
    logging.info("Found {count} products in {category} with {feature}".format(count=cursor.count(),
                                                                              category=category_id,
                                                                              feature=feature_name))
    job_results = []
    for prod in cursor:
        res = download_images_q.enqueue(download_image, prod, feature_name, category_id, max_images)
        job_results.append(res.result)
        print('results are:' + str(res.result))
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
                    logging.warning("Could not save image")
                    return 0
                # downloaded_images += 1
                logging.info("Saved... Downloaded approx. {0} images in this category/feature combination"
                             .format(downloaded_images))
                return 1
            else:
                # TODO: Count number of irrelevant images (for statistics)
                return 0

def run():
    logging.info('Starting...')

    # Leftovers:
    descriptions = ['round collar', 'bow collar',
                    'ribbed round neck', 'rollneck',
                    'slash neck']

    descriptions_dict = {'bowcollar': "\"bow collar\" bowcollar",
                         'crewneck': "\"crew neck\" \"crew neckline\" crewneck \"classic neckline\"",
                         'roundneck': "\"round neck\" \"round neckline\" roundneck",
                         'scoopneck': "\"scoopneck\" \"scoop neckline\" scoopneck",
                         'squareneck': "\"square neck\" \"square neckline\" squareneck",
                         'v-neck': "\"v-neck\" \"v-neckline\"  \"v neckline\" vneck"}

    job_results_dict = dict.fromkeys(descriptions_dict)

    for name, search_string in descriptions_dict.iteritems():
        job_results = find_and_download_images(name, search_string, "dresses", MAX_IMAGES)
        job_results_dict[name] = job_results

    while True:
        for name, jrs in job_results_dict.iteritems():
            logging.info(
                "{0}: Downloaded {1} images...".format(name, sum((done.result for done in jrs if done.result))))

def print_logging_info(msg):
    print msg

# hackety hack
logging.info = print_logging_info

if __name__ == '__main__':
    run()
