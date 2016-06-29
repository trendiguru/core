'''
backend funtions for neurodoll_mask webapp
'''
from ..constants import db, ultimate_21_dict
from ..paperdoll import neurodoll_falcon_client as nfc
from ..Utils import get_cv2_img_array
import numpy as np
import cv2
from random import randint

def grabcut_neuro(img_url, neuro_mask, fg, sortOrder):
    image = get_cv2_img_array(img_url)
    if image is None:
        print ('bad img url')
        return False, []

    # rect = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[neuro_mask>200*fg]=2
    mask[neuro_mask>255*fg]=1
    mask[neuro_mask <200 * fg] = 3
    mask[neuro_mask <55 * fg] = 0
    cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, 3, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
    image[mask2==0]=0
    filename = '/var/www/neuro_mask/grabcut_' + str(sortOrder) + '.jpg'
    cv2.imwrite(filename, image)
    return True, mask2


def middleman(imgs, category, fg=0.75, neurodoll=True):
    category_idx = ultimate_21_dict[category]
    masks = []
    for img in imgs:
        sortOrder = img['sortOrder']
        url = 'http:'+img['itemImgUrl']
        if neurodoll:
            dic = nfc.pd(url, category_idx)
            if not dic['success']:
                tmp = {'sortOrder': sortOrder,
                       'itemImgUrl': url,
                       'success':False}
                masks.append(tmp)
                continue
            mask = dic['mask']
            filename = '/var/www/neuro_mask/neuro_' + str(sortOrder) + '.jpg'
            cv2.imwrite(filename, mask)
        else:
            filename = '/var/www/neuro_mask/neuro_' + str(sortOrder) + '.jpg'
            mask = cv2.imread(filename,0)

        success, grabcut_mask = grabcut_neuro(url, mask,fg, sortOrder)
        if not success:
            tmp = {'sortOrder': sortOrder,
                   'itemImgUrl': url,
                   'success': False}
            masks.append(tmp)
            continue
        tmp = {'sortOrder':sortOrder,
               'itemImgUrl': url,
               'success': True}

        masks.append(tmp)
    return masks

def fresh_meat(gender,item_id, fg=0.75):
    col_name= 'recruit_'+gender
    collection = db[col_name]
    do_neuro = True

    if item_id:
        exists = collection.find_one({'id': item_id})
        if exists:
            do_neuro = False
            imgs = exists['raw_info']['itemImgInfoList']
            category = exists['categories']

    if do_neuro:
        r = randint(0,1000)
        item =  collection.find({'categories':{'$in':ultimate_21_dict.keys()}})[r]
        imgs = item['raw_info']['itemImgInfoList']
        item_id = item['id'][0]
        category = item['categories']

    masks = middleman(imgs, category,fg, do_neuro)
    ret = {'item_id': item_id,
           'category': category,
           'mask': masks}
    return ret


