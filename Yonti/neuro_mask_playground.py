'''
backend funtions for neurodoll_mask webapp
'''
from ..constants import db, ultimate_21_dict
from ..paperdoll import neurodoll_falcon_client as nfc
from ..Utils import get_cv2_img_array
import numpy as np
import cv2


def grabcut_neuro(img_url, neuro_mask, fg, bg):
    image = get_cv2_img_array(img_url)
    if image is None:
        print ('bad img url')
        return False, []

    rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[neuro_mask<255*fg]=2
    mask[neuro_mask<255*fg]=3
    cv2.grabCut(image, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    return True, mask2


def middleman(col_name,idx, imgs, category, fg=0.75, neurodoll=True):
    category_idx = ultimate_21_dict[category]
    collection = db[col_name]
    masks = []
    for img in imgs:
        sortOrder = img['sortOrder']
        url = 'http:'+img['itemImgUrl']
        if neurodoll:
            dic = nfc.pd(url, category_idx)
            if not dic['success']:
                continue
            mask = dic['mask']
        else:
            mask = img['neuro_mask']
        success, grabcut_mask = grabcut_neuro(url, mask)
        if not success:
            continue
        bg = 1- fg
        tmp = {'sortOrder':sortOrder,
               'itemImgUrl': url,
               'neuro_mask': mask,
               'fg': fg,
               'bg': bg,
               'grabcut_mask': grabcut_mask}
        masks.append(tmp)
    collection.update_one({'_id': idx},{'$set':{'images.processed' : masks}})
    return masks

def fresh_meat(gender):
    col_name= 'recruit_'+gender
    collection = db[col_name]
    item =  collection.find_one({'images.processed': {'$exists': 0}})
    imgs = item['raw_info']['itemImgInfoList']
    item_id = item['_id']
    category = item['categories']
    masks = middleman(col_name, item_id, imgs, category)
    ret = {'item_id': item_id,
           'imgs': imgs,
           'category': category,
           'mask': masks}
    return ret


def grabcut_only(item_id,gender, fg):
    col_name= 'recruit_'+gender
    collection = db[col_name]
    id_exists =  collection.find_one({'id': item_id})
    if not id_exists:
        ret = fresh_meat(gender)
        return ret
    mask_exists = collection.find_one({'id': item_id, 'images.processed':{'$exists':1}})
    if mask_exists:
        do_neuro = False
    else:
        do_neuro = True
    imgs = mask_exists['images']['precessed']
    category = mask_exists['categories']
    masks = middleman(col_name, item_id, imgs, category, fg, do_neuro)
    ret = {'item_id': item_id,
           'imgs': imgs,
           'category': category,
           'mask': masks}

    return ret