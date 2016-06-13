from . import constants

from constants import db


# ------------------------------------------------ IMAGE-LEVEL ---------------------------------------------------------

def cancel(image_id, person_id, item_id):
    image_obj = db.images.find_one({'_id': image_id})
    if not image_obj:
        return False
    if item_id:
        # CANCEL ITEM
        res = db.images.update_one({'_id': image_id}, {'$pull': {'people': {'items': {'$elemMatch': {'item_id': item_id}}}}})
        return bool(res.modified_count)
    elif person_id:
        # CANCEL PERSON
        res = db.images.update_one({'_id': image_id}, {'$pull': {'people': {'person_id': person_id}}})
        return bool(res.modified_count)
    else:
        # CANCEL IMAGE (INSERTY TO IRRELEVANT_IMAGES BEFORE)
        sparse_obj = shrink_image_object(image_obj)
        db.irrelevant_images.insert_one(sparse_obj)
        db.images.delete_one({'_id': image_id})
        return True


# ----------------------------------------------- PERSON-LEVEL ---------------------------------------------------------



# ------------------------------------------------ ITEM-LEVEL ----------------------------------------------------------



# ----------------------------------------------- CO-FUNCTIONS ---------------------------------------------------------

def shrink_image_object(image_obj):
    image_obj['relevant'] = False
    image_obj['people'] = []
    image_obj.pop('num_of_people')
    image_obj.pop('image_id')
    return image_obj
