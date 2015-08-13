import os
import logging
import cv2
import pymongo
import Utils
import background_removal
from find_similar_mongo import get_all_subcategories

db = pymongo.MongoClient().mydb

MAX_IMAGES = 10000

def find_images(feature_name, search_string, category_id, max_images):

    logging.debug('starting to find ' + feature_name)

    #TODO: search in subcategories as well
    query = {"description": {'$text': search_string},
             "categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    fields = {"categories": 1, "image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1,
              "id": 1, "description": 1, "feature_bbs": 1}

    downloaded_images = 0

    cursor = db.products.find(query, fields)
    logging.debug("Found {count} products in {category} with {feature}".format(count=cursor.count(),
                                                                               category=category_id,
                                                                               feature=feature_name))

    for prod in cursor:
        if downloaded_images < max_images:
            xlarge_url = prod['image']['sizes']['XLarge']['url']

            img_arr = Utils.get_cv2_img_array(xlarge_url)
            if img_arr is None:
                return None

            relevance = background_removal.image_is_relevant(img_arr)

            if relevance.is_relevant:

                filename = os.path.join('{0}{1}'.format(feature_name, category_id), feature_name)
                Utils.ensure_dir(filename)
                filename = os.path.join(feature_name, str(prod["id"]) + '.jpg')

                cv2.imwrite(filename, img_arr)
                downloaded_images += 1
                logging.debug("Downloaded {0}".format(downloaded_images))
            else:
                # TODO: Count number of irrelevant images (for statistics)
                pass




if __name__ == '__main__':
    print('Starting...')

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

    for name, search_string in descriptions_dict.iteritems():
        find_images(name, search_string, "dresses", MAX_IMAGES)
