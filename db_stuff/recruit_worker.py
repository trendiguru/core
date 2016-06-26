from .recruit_constants import recruitID2generalCategory, api_stock
from time import time
import requests
import json
from ..constants import db, fingerprint_version, redis_conn
from datetime import datetime
import logging
from rq import Queue

q = Queue('recruit_worker', connection=redis_conn)
today_date = str(datetime.date(datetime.now()))


def log2file(LOG_FILENAME='/home/developer/yonti/recruit_download_stats.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILENAME, mode= 'a')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def GET_ByGenreId( genreId, page=1,limit=1, instock = False):
    res = requests.get('http://itemsearch.api.ponparemall.com/1_0_0/search/'
                       '?key=731d157cb0cdd4146397ef279385d833'
                       '&genreId='+genreId +
                       '&format=json'
                       '&limit='+str(limit) +
                       '&page='+str(page) +
                       '&inStockFlg='+str(int(instock)))
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
        img_url = []
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
                   "raw_info": item}

        # image = Utils.get_cv2_img_array(img_url)
        # if image is None:
        #     print ('bad img url')
        #     continue

        # img_hash = get_hash(image)
        #
        # hash_exists = collection.find_one({'img_hash': img_hash})
        # if hash_exists:
        #     print ('hash already exists')
        #     continue
        #
        # generic["img_hash"] = img_hash

        collection.insert_one(generic)
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
        q.enqueue(genreDownloader, args=(genreId, end_page), timeout=5400)
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