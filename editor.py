from . import constants

from constants import db


# ------------------------------------------------ IMAGE-LEVEL ---------------------------------------------------------

def cancel_image(image_url):
    image_obj = db.images.find_one({'image_urls': image_url})
    if image_obj:
        # sparse_obj = shrink object
        db.irrelevant_images.insert_one(image_obj)
        db.images.delete_many({'image_urls': image_url})
        return True
    else:
        return False



# ----------------------------------------------- PERSON-LEVEL ---------------------------------------------------------


# ------------------------------------------------ ITEM-LEVEL ----------------------------------------------------------
