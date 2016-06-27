from .ebay_constants import ebay_account_info, ebay_gender, categories_badwords, \
    categories_keywords, ebay_paperdoll_women
from time import time, sleep
import requests
import json
from ..constants import db, fingerprint_version, redis_conn
from datetime import datetime
import logging
from rq import Queue
import re
q = Queue('ebay_API_worker', connection=redis_conn)
today_date = str(datetime.date(datetime.now()))


def log2file(log_filename, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename, mode= 'a')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def GET_call(GEO, gender, sub_attribute, price_bottom=0, price_top=10000, page=1, num=100):
    account_info = ebay_account_info[GEO]
    gender_attribute = ebay_gender[gender]
    price_attribute = 'price_range_' + str(price_bottom) + '_' + str(price_top)
    api_call =  'http://sandbox.api.ebaycommercenetwork.com/publisher/3.0/json/GeneralSearch?' \
                'apiKey='+account_info['API_Key'] + \
                '&visitorUserAgent=""' \
                '&visitorIPAddress=""' \
                '&trackingId='+account_info['Tracking_ID'] + \
                '&categoryId=31515' \
                '&pageNumber=' + str(page) + \
                '&numItems=' + str(num) + \
                '&attributeValue=' + gender_attribute + \
                '&attributeValue=' + sub_attribute + \
                '&attributeValue=' + price_attribute
    # 'showProductOffers = true' \
    # '&numOffersPerProduct=1' \

    res = requests.get(api_call)

    if res.status_code != 200:
        return False, 0, []

    dic = json.loads(res.text)
    categories = dic['categories']['category']
    if not len(categories) or 'items' not in categories[0].keys():
        return True, 0, []

    items = categories[0]['items']
    item = items['item']
    if not len(item):
        return True, 0, []

    total_item_count= int(items['matchedItemCount'])
    return True, total_item_count, item


def fromCats2ppdCats(gender, cats, sub_attribute):
    ppd_cats = []
    for cat in cats:
        ppd_cats.append(ebay_paperdoll_women[cat])
    cat_count = len(ppd_cats)
    #TODO: use sub_attribute
    if cat_count>1:
        if 'shorts' in ppd_cats:
            ppd_cats.remove('shorts')
        if 'polo' in ppd_cats:
            cat = 'polo'
        elif 't-shirt' in ppd_cats:
            cat = 't-shirt'
        elif 'shirt' in ppd_cats:
            cat = 'shirt'
        elif 'blazer' in ppd_cats:
            cat = 'blazer'
        elif 'bikini' in ppd_cats:
            cat = 'bikini'
        elif 'swimsuit' in ppd_cats:
            cat =  'swimsuit'
        elif 'dress' in ppd_cats:
            cat = 'dress'
        elif 'sweater' in ppd_cats:
            cat = 'sweater'
        elif 'sweatshirt' in ppd_cats:
            cat =  'sweatshirt'
        elif 'pants' in ppd_cats:
            cat =  'pants'
        elif 'belt' in ppd_cats:
            cat = 'belt'
        else:
            cat = ppd_cats[0]
    elif cat_count == 0:
        print ('count = 0 for %s' % ppd_cats)
        return []
    else:
        cat = ppd_cats[0]

    if cat in ['dress', 'bikini', 'stocking', 'leggings', 'skirt'] and gender is 'Male':
        print ('Male X female cat')
        return []

    return cat


def find_keywords(desc):
    DESC = desc.upper()
    split1 = re.split(' |-|,|;|:', DESC)
    cats = []

    if any(x in DESC for x in ['BELT BUCKLE', 'BELT STRAP']):
        return False, []
    for s in split1:
        if s in categories_keywords:
            cats.append(s)
        elif s in categories_badwords:
            # print ('%s in badwords' % s)
            return False, []
        else:
            pass

    return True, cats


def name2category(gender, name, sub_attribute, desc):
    status, cats = find_keywords(name)
    if not status:
        return False, []

    if not len(cats):
        print ('%s not in keywords' % name)
        if len(desc)>0:
            status, cats = find_keywords(desc)
            if not status:
                return False, []
        if not len(cats):
            logger_keywords = log2file('/home/developer/yonti/keywords_' + gender + '.log', 'keyword')
            logger_keywords.info(name)
            return False, []

    ppd_cats = fromCats2ppdCats(gender,cats, sub_attribute)
    if len(ppd_cats):
        return True, ppd_cats
    else:
        return False, []


def process_items(items, gender,GEO , sub_attribute):
    col_name = 'ebay_'+gender+'_'+GEO
    collection = db[col_name]
    new_items = 0
    for item in items:
        offer = item['offer']
        keys = offer.keys()
        itemId = offer['id']
        sku = offer['sku']
        id_exists = collection.find_one({'id': itemId})
        sku_exists = collection.find_one({'sku': sku})
        if id_exists :
            #TODO: add checks
            # print ('ID ID ID ID')
            continue

        if sku_exists:
            # TODO: add checks
            print ('SKU SKU SKU')
            continue

        price = {'price': offer['basePrice']['value'],
                 'currency': offer['basePrice']['currency']}

        offer_status = offer['stockStatus']
        if offer_status == 'in-stock':
            status = {"instock": True, "days_out": 0}
        else:
            status = {"instock": False, "days_out": 0}

        img_list = offer['imageList']['image']
        '''
        take highest res img
        '''
        img_url = img_list[-1]

        if 'description' in keys:
            desc = offer['description']
        else:
            desc = ""

        if 'shippingCost' in keys:
            shipping = offer['shippingCost']
        else:
            shipping = ""

        success, category = name2category(gender, offer['name'], sub_attribute, desc)
        if not success:
            # print ('NOT SUCCESS NOT SUCCESS')
            continue
        if 'manufacturer' in keys:
            brand = offer['manufacturer']
        else:
            brand = offer['store']['name']

        generic = {"id": [itemId],
                   "categories": category,
                   "clickUrl": offer['offerURL'],
                   "images": {"XLarge": img_url},
                   "status": status,
                   "shortDescription": offer['name'],
                   "longDescription": desc,
                   "price": price,
                   "brand": brand,
                   "download_data": {'dl_version': today_date,
                                     'first_dl': today_date,
                                     'fp_version': fingerprint_version},
                   "fingerprint": None,
                   "gender": gender,
                   "shippingInfo": shipping,
                   "raw_info": offer}

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
    return new_items, len(items)


def downloader(GEO, gender, sub_attribute, price_bottom=0, price_top=10000):
    start_time = time()
    sleep(1)
    success, item_count, items = \
        GET_call(GEO, gender, sub_attribute, price_bottom, price_top, num=1)

    if not success:
        print ('GET failed')
        return

    if not item_count:
        print ('no results found for %s in the price range of %d to %d'
               %(sub_attribute, price_bottom, price_top))
        return

    if item_count > 1499 and price_top > price_bottom:
        middle = int((price_top+price_bottom)/2)
        if middle >= price_bottom:
            q.enqueue(downloader, args=(GEO, gender, sub_attribute, price_bottom, middle), timeout=5400)
        if price_top > middle != price_bottom:
            q.enqueue(downloader, args=(GEO, gender, sub_attribute, middle, price_top), timeout=5400)
        print ('price range %d to %d divided to %d - %d  &  %d - %d'
               % (price_bottom, price_top, price_bottom, middle, middle, price_top))
        return

    end_page = item_count/100 +2
    if end_page == 17:
        end_page = 16
    new_items = total_items = 0
    # new_inserts, total = process_items(items, gender, GEO, sub_attribute)
    # new_items += new_inserts
    # total_items += total
    for i in range(1, end_page):
        sleep(1)
        success, item_count, items = GET_call(GEO, gender, sub_attribute, price_bottom, price_top, page=i, num=100)
        if not success:
            continue
        new_inserts, total = process_items(items, gender, GEO, sub_attribute)
        new_items += new_inserts
        total_items += total
    end_time = time()
    logger = log2file('/home/developer/yonti/ebay_'+gender+'_download_stats.log', 'download')
    summery = 'attribute: %s_%s ,price: %d to %d , item Count: %d, new: %d, download_time: %d' \
              % (gender, sub_attribute, price_bottom, price_top, total_items, new_items, (end_time-start_time))
    logger.info(summery)
    print(summery)