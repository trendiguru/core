from .recruit_constants import recruitID2generalCategory, api_stock, recruit2category_idx
from time import time, sleep
import requests
import json
from ..constants import db, fingerprint_version, redis_conn
from datetime import datetime
import logging
from rq import Queue
from ..Utils import get_cv2_img_array
import hashlib
from ..fingerprint_core import generate_mask_and_insert
import re
recruit_q = Queue('recruit_worker', connection=redis_conn)
fp_q = Queue('fingerprint_new', connection=redis_conn)

today_date = str(datetime.date(datetime.now()))


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def log2file(LOG_FILENAME='/home/developer/yonti/recruit_download_stats.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILENAME, mode= 'a')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def GET_ByGenreId( genreId, page=1,limit=1, img_size=500, instock = False):
    res = requests.get('http://itemsearch.api.ponparemall.com/1_0_0/search/'
                       '?key=731d157cb0cdd4146397ef279385d833'
                       '&genreId='+genreId +
                       '&format=json'
                       '&limit='+str(limit) +
                       '&page='+str(page) +
                       '&inStockFlg='+str(int(instock)) +
                       '&imgSize=' + str(img_size))
    if res.status_code != 200:
        return False, []
    dic = json.loads(res.text)
    if 'itemInfoList' in dic.keys():
        return True, dic
    else:
        return False, []


def process_items(item_list, gender,category):
    col_name = 'recruit_'+gender
    collection = db[col_name]
    new_items = 0
    for item in item_list:
        itemId = item['itemId']
        exists = collection.find_one({'id': itemId})
        if exists:
            #TODO: add checks
            # print ('item already exists')
            continue

        price = {'price': item['salePriceIncTax'],
                 'currency': 'Yen'}

        status = 'instock'
        image = None
        imgs = item['itemImgInfoList']
        for i in range(len(imgs)):
            img =imgs[i]['itemImgUrl']
            split1 = re.split(r'\?|&', img)
            if 'ver=1' not in split1:
                continue
            img_url = 'http:' + img
            image = get_cv2_img_array(img_url)
            if image is not None:
                break

        if image is None:
            print ('bad img url')
            continue

        img_hash = get_hash(image)

        hash_exists = collection.find_one({'img_hash': img_hash})
        if hash_exists:
            print ('hash already exists')
            continue

        if 'itemDescriptionText' in item.keys():
            description = item['itemDescriptionText']
        else:
            description = []
        generic = {"id": [itemId],
                   "categories": category,
                   "clickUrl": item['itemUrl'],#TODO: add tracking_id
                   "images": {"XLarge": img_url},
                   "status": status,
                   "shortDescription": item['itemName'],
                   "longDescription": description,
                   "price": price,
                   "brand": item['shopName'],
                   "download_data": {'dl_version': today_date,
                                     'first_dl': today_date,
                                     'fp_version': fingerprint_version},
                   "fingerprint": None,
                   "gender": gender,
                   "shippingInfo": [],
                   "raw_info": item,
                   "img_hash": img_hash}


        # collection.insert_one(generic)
        while fp_q.count>5000:
            sleep(30)

        fp_q.enqueue(generate_mask_and_insert, doc=generic, image_url=img_url,
                  fp_date=today_date, coll=col_name, img=image, neuro=True)
        new_items+=1
    return new_items, len(item_list)


def genreDownloader(genreId, start_page=1):
    start_time = time()
    success, response_dict = GET_ByGenreId(genreId, page=start_page, limit=100, instock=True)
    if not success:
        print ('GET failed')
        return
    if genreId[1] == '1':
        gender = 'Female'
    else:
        gender = 'Male'
    new_items = total_items = 0
    category = recruitID2generalCategory[genreId]
    sub = [x for x in api_stock if x['genreId'] == genreId][0]['category_name']
    new_inserts, total = process_items(response_dict["itemInfoList"], gender, category)
    new_items += new_inserts
    total_items += total
    end_page = int(response_dict['pageCount'])
    if end_page-start_page > 25:
        end_page = start_page+25
        if end_page>999:
            end_page=999
        while recruit_q.count>50:
            sleep(30)
        recruit_q.enqueue(genreDownloader, args=(genreId, end_page), timeout=5400)
    for i in range(start_page+1, end_page):
        success, response_dict = GET_ByGenreId(genreId, page=i, limit=100, instock=True)
        if not success:
            continue
        new_inserts, total = process_items(response_dict["itemInfoList"], gender, category)
        new_items += new_inserts
        total_items += total
    end_time = time()
    logger = log2file()
    summery = 'genreId: %s,start_page: %d, end_page: %d Topcategory: %s, Subcategory: %s, total: %d, new: %d, download_time: %d' \
              % (genreId, start_page, end_page,category, sub, total_items, new_items, (end_time-start_time))
    logger.info(summery)
    print(sub + ' Done!')

def deleteDuplicates(delete=True):
    '''
    true for deleting
    false for only printing out
    '''
    for gender in ['Male','Female']:
        col = db['recruit_'+gender]
        print ('\n #### %s ######' % gender)
        for cat in recruit2category_idx.keys():
            items = col.find({'categories':cat})
            count = items.count()
            for item in items:
                idx1 = item['_id']
                item_id = item['id']
                img_url = item['images']['XLarge']
                exists = col.find({'categories':cat, 'images.XLarge':img_url})
                if exists:
                    idx2 = exists[0]['_id']
                    item_id2 = exists[0]['id']
                    if exists.count()==1 and idx1 == idx2 and item_id == item_id2 :
                        continue

                    print ("url = %s , _id = %s , item_id = %s , butURL = %s" %(img_url, idx1, item_id, item['clickUrl']))
                    print ('dups:')
                    for e in exists:
                        if idx1 == e['_id']:
                            continue
                        dup = e['images']['XLarge']
                        print ("url = %s , _id = %s , item_id = %s , buyURL = %s" % (dup, e['_id'], e['id'], e['clickUrl']))
                    raw_input()

