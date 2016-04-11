import numpy as np
from . import background_removal
from . import constants
from . import Utils
from . import kassper
from gcloud import storage
from oauth2client.client import GoogleCredentials

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
    # To upload from 2d-mask
    blb.upload_from_string(data.tostring())


def create_training_set_with_grabcut():
    i = 1
    total = db.training_images.count()
    for doc in db.training_images.find():
        if not i % 1000:
            print "did {0}/{1} documents".format(i, total)
        url = doc['url'].split('/')[-1]
        img_url = 'https://tg-training.storage.googleapis.com/tamara_berg_street2shop_dataset/images/' + url
        image = Utils.get_cv2_img_array(img_url)
        if image is None:
            print "{0} is a bad image".format(img_url)
            continue
        skin_mask = kassper.skin_detection_with_grabcut(image, image, skin_or_clothes='skin')
        mask = np.where(skin_mask == 255, 1, 0).astype(np.uint8)
        for item in doc['items']:
            item_bb = get_margined_bb(image, item['bb'], 0.1)
            category_num = cats.index(item['category'])
            item_mask = background_removal.simple_mask_grabcut(image, rect=item_bb)
            mask = np.where(item_mask == 255, category_num, mask)
        filename = 'tamara_berg_street2shop_dataset/masks/' + url[:-4]
        save_to_storage(bucket, mask, filename)
        db.training_images.update_one({'_id': doc['_id']}, {'$set': {'mask_url': filename}})
