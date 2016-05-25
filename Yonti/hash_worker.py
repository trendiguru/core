'''
blue colour worker
'''

import hashlib
from .. import Utils, constants
import sys
db = constants.db

def get_hash(collection_name,item_count,item_id,item_url):
    collection = db[collection_name]

    image = Utils.get_cv2_img_array(item_url)
    if image is None:
        print('bad image!')
        collection.delete_one({"_id":item_id})
        return

    m = hashlib.md5()
    m.update(image)
    img_hash = m.hexdigest()
    collection.update_one({"_id":item_id},{"$set":{'img_hash':img_hash}})
    print ('item %d updated' %(item_count))
    return