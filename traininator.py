import numpy as np
import time
import cv2
from trendi import background_removal
from trendi import constants
from trendi import Utils
from gcloud import storage
from oauth2client.client import GoogleCredentials
import sys

import urllib2



db = constants.db
cats = constants.tamara_berg_categories

credentials = GoogleCredentials.get_application_default()
bucket = storage.Client(credentials=credentials).bucket("tg-training")


def get_margined_bb(image, bb, buffer):
    x, y, w, h = bb
    x_back = np.max((x - int(buffer*w), 0))
    x_ahead = np.min((x + int((1+buffer)*w), image.shape[1]-1))
    y_up = np.max((y - int(buffer*h), 0))
    y_down = np.min((y + int((1+buffer)*h), image.shape[0]-1))
    return [x_back, y_up, x_ahead-x_back, y_down-y_up]


def save_to_storage(buck, data, filename):
    blb = buck.blob(filename)
    mask_str = cv2.imencode('.png', data)[1].tostring()
    # To upload from 2d-mask
    blb.upload_from_string(mask_str)


def create_training_set_with_grabcut(collection):
    coll = db[collection]
    i = 1
    total = db.training_images.count()
    start = time.time()
    for doc in coll.find():
        if not i % 10:
            print "did {0}/{1} documents in {2} seconds".format(i, total, time.time()-start)
            print "average time for image = {0}".format((time.time()-start)/i)
        url = doc['url'].split('/')[-1]
        img_url = 'https://tg-training.storage.googleapis.com/tamara_berg_street2shop_dataset/images/' + url
        image = Utils.get_cv2_img_array(img_url)
        if image is None:
            print "{0} is a bad image".format(img_url)
            continue
        i += 1
        small_image, ratio = background_removal.standard_resize(image, 600)
        # skin_mask = kassper.skin_detection_with_grabcut(small_image, small_image, skin_or_clothes='skin')
        # mask = np.where(skin_mask == 255, 35, 0).astype(np.uint8)
        mask = np.zeros(small_image.shape[:2], dtype=np.uint8)
        for item in doc['items']:
            try:
                bb = [int(c/ratio) for c in item['bb']]
                item_bb = get_margined_bb(small_image, bb, 0)
                if item['category'] not in cats:
                    continue
                category_num = cats.index(item['category'])
                item_mask = background_removal.simple_mask_grabcut(small_image, rect=item_bb)
            except:
                continue
            mask = np.where(item_mask == 255, category_num, mask)
        filename = 'tamara_berg_street2shop_dataset/masks/' + url[:-4] + '.txt'
        save_to_storage(bucket, mask, filename)
        coll.update_one({'_id': doc['_id']}, {'$set': {'mask_url': 'https://tg-training.storage.googleapis.com/' + filename}})
    print "Done masking! took {0} seconds".format(time.time()-start)

def bucket_to_training_set(collection):
    coll = db[collection]
    i = 1
    total = db.training_images.count()
    print(str(total)+' images in collection '+collection)
    start = time.time()
    for i in range(0,500000):
        photo_name = 'photo_'+str(i)+'.jpg'
        img_url = 'https://tg-training.storage.googleapis.com/tamara_berg_street2shop_dataset/images/'+photo_name
        print('\nattempting to open '+img_url)
        try:
            ret = urllib2.urlopen(img_url)
            if ret.code == 200:
                print(photo_name+" exists, checking if in db")
                doc = coll.find({'url','/home/jeremy/dataset/images/'+photo_name})
                print('doc '+str(doc))
                if doc :
                    print('found doc for '+str(photo_name)+' in db already')
                else:
                    print('doc for '+str(photo_name)+' not found, add to db')

            else:
                print('image '+photo_name +' not found')
        except:
            print('error trying to open '+photo_name+' err:'+str(sys.exc_info()[0]))


if __name__ == "__main__":
    bucket_to_training_set('training_images')
