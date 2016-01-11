
from trendi.constants import db
from trendi import constants
from trendi import Utils
import numpy as np

from trendi.paperdoll import paperdoll_parse_enqueue
    #import background_removal
from trendi import paperdolls


training_collection_cursor = db.training2.find()
assert (training_collection_cursor)  # make sure training collection exists

pruned_images=[]
c = 0
while c<2: # < num of doc items (collection items)
    doc = next(training_collection_cursor, None)
    images = doc['images']
    for img in images:
       #     if Utils.good_bb(img, skip_if_marked_to_skip=True) and good_img(img):
        pruned_images.append(img["url"])
        print(pruned_images)
        print(len(pruned_images))
    c+=1

img_arrs = []
masks = []
mask_items = []
for url1 in pruned_images:
    max_retry = 5
    got_mask = False
    img_arr = Utils.get_cv2_img_array(url1)
    while max_retry and not got_mask:
        try:
            mask, labels, pose = paperdoll_parse_enqueue.paperdoll_enqueue(img_arr, at_front=True, async=False).result[:3]
            got_mask = np.any(mask) # condition for legal mask?
            print(str(got_mask))
            masks.append(mask)

            final_mask = paperdolls.after_pd_conclusions(mask, labels)#, person['face'])
            for num in np.unique(final_mask):
                category = list(labels.keys())[list(labels.values()).index(num)]
                if category == 'dress'  and category in constants.paperdoll_shopstyle_women.keys():
                    print("Found dress!!")
                    mask_item = 255 * np.array(final_mask == num, dtype=np.uint8)
                    mask_items.append(mask_item)

        except Exception as e:
            max_retry = max_retry - 1
            print("Failed to get mask with exception:")
            print e.message

    if not got_mask:
        print url1 + " failed."
        continue
    else:
        print "Success: " + url1


np.save("yuli_masks.npy", masks)
np.save("yuli_mask_items.npy", mask_items)
