from trendi import constants
from trendi.constants import db
from trendi import Utils
import numpy as np

from trendi.paperdoll import paperdoll_parse_enqueue
    #import background_removal
from trendi import paperdolls


training_collection_cursor = db.training2.find()
assert (training_collection_cursor)  # make sure training collection exists
doc = next(training_collection_cursor, None)

pruned_images=[]
while doc is not None:
    images = doc['images']
    for img in images:
       #     if Utils.good_bb(img, skip_if_marked_to_skip=True) and good_img(img):
            pruned_images.append(img["url"])

img_arrs=[]
masks = []
mask_items = []
for url1 in pruned_images:
    img_arrs.append( Utils.get_cv2_img_array(url1, convert_url_to_local_filename=True, download=True))
    mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(img, async=False).result[:3]
    masks.append(mask)
    print("Mask shape: "+mask.shape)
    final_mask = paperdolls.after_pd_conclusions(mask, labels)#, person['face'])
    for num in np.unique(final_mask):
        category = list(labels.keys())[list(labels.values()).index(num)]
        if category == 'dress'  and category in constants.paperdoll_shopstyle_women.keys():
            print("Found dress!!")
            mask_item = 255 * np.array(final_mask == num, dtype=np.uint8)
            mask_items.append(mask_item)

np.save("yuli_masks.npy", masks)
np.save("yuli_mask_items.npy", mask_items)
