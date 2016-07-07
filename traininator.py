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

#TO UPLOAD IMAGES TO BUCKET:
# gsutil -m cp -r new_photos_512x512/ gs://tg-training/tamara_berg_street2shop_dataset/images

# TO GRANT PERMISSION TO WORLD TO SEE:
#gsutil acl ch -u AllUsers:R gs://tg-training/tamara_berg_street2shop_dataset/images/*

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
    '''
    Takes a bucket of data and adds to db collection
    if not in db, add
    if in db, fix url, make user a list, already_done is counter

    :param collection: mongodb colleciton
    :return:
    '''
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
                try:
                    doc = coll.find({'url':'/home/jeremy/dataset/images/'+photo_name})
                    if doc :
                        print('found doc for '+str(photo_name)+' in db already')
                        doc = doc[0]
                        print(doc)
                        id = None
                        already_done = None
                        already_done_image_level = None
                        already_seen_image_level = None
                        user_name = None
                        if '_id' in doc:
                            id = doc['_id']
                        if 'already_done' in doc:
                            already_done = doc['already_done']
                            del doc['already_done']
                            doc['already_seen_image_level'] = 1
                        if 'already_seen_image_level' in doc:
                            already_seen_image_level = doc['already_seen_image_level']
                            doc['already_seen_image_level'] = 1
                        if 'user_name' in doc:
                            user_name = doc['user_name']
                            if isinstance(user_name,basestring):
                                doc['user_name'] = [user_name]
                        url = doc['url']
                        doc['url'] = img_url
                        print('id {} ad {} asil {} un {}'.format(id,already_done,already_seen_image_level,user_name))
                        print('items:'+str(doc['items']))
                        print('new doc:\n'+str(doc))
#                        res = coll.replace_one({'_id':id},doc)

                    else:
                        print('doc for '+str(photo_name)+' not found, add to db')
                except:
                    print('error trying to get doc , err:'+str(sys.exc_info()[0]))

            else:
                print('image '+photo_name +' not found (ret code not 200)')
        except:
            print('error trying to open '+photo_name+' err:'+str(sys.exc_info()[0]))
   #     raw_input('ret to cont')

if __name__ == "__main__":
    bucket_to_training_set('training_images')
