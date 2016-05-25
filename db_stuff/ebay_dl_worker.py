from time import sleep,time
from StringIO import StringIO
import gc
import gzip
from . import ebay_dl_utils
from . import ebay_constants
import re
import csv
import hashlib
from .. import constants, Utils
from ..fingerprint_core import generate_mask_and_insert
from rq import Queue
import datetime
import sys
maxInt = sys.maxsize
decrement = True
while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

today_date = str(datetime.datetime.date(datetime.datetime.now()))

db = constants.db
q = Queue('fingerprint_new', connection=constants.redis_conn)


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def ebay2generic(item, gender, subcat):
    try:
        generic = {"id": item["\xef\xbb\xbfOFFER_ID"],
                   "categories": subcat,
                   "clickUrl": item["OFFER_URL_MIN_CATEGORY_BID"],
                   "images": {"XLarge": item["IMAGE_URL"]},
                   "status": {"instock" : True, "days_out" : 0},
                   "shortDescription": item["OFFER_TITLE"],
                   "longDescription": item["OFFER_DESCRIPTION"],
                   "price": {'price': item["PRICE"],
                             'priceLabel': "USD"   },
                   "Brand" : item["MANUFACTURER"],
                   "Site" : item["MERCHANT_NAME"],
                   "download_data": {'dl_version': today_date,
                                     'first_dl': today_date,
                                     'fp_version': constants.fingerprint_version},
                   "fingerprint": None,
                   "gender": gender,
                   "ebay_raw": item}
        if item["STOCK"] != "In Stock":
            generic["status"]["instock"] = False
        image = Utils.get_cv2_img_array(item["IMAGE_URL"])
        if image is None:
            generic = None
        else:
            img_hash = get_hash(image)
            generic["img_hash"] = img_hash

    except:
        print item
        generic = None
    return generic

def fromCats2ppdCats(gender, cats):
    ppd_cats = []
    for cat in cats:
        ppd_cats.append(ebay_constants.ebay_paperdoll_women[cat])
    cat_count = len(ppd_cats)
    if gender == '':
        gender = 'Unisex'
    if cat_count>1:
        if 'polo' in ppd_cats:
            cat = 'polo'
        elif 't-shirt' in ppd_cats:
            cat = 't-shirt'
        elif 'shirt' in ppd_cats:
            cat = 'shirt'
        elif 'blazer' in ppd_cats:
            cat = 'blazer'
        elif 'bikini' in ppd_cats:
            cat =  'bikini'
        elif 'swimsuit' in ppd_cats:
            cat =  'swimsuit'
        elif 'sweater' in ppd_cats:
            cat = 'sweater'
        elif 'sweatshirt' in ppd_cats:
            cat =  'sweatshirt'
        elif 'pants' in ppd_cats:
            cat =  'pants'
        elif 'belt' in ppd_cats:
            cat = 'belt'
        elif 'dress' in ppd_cats and gender == 'Male':
            ppd_cats.remove('dress')
            cat = ppd_cats[0]
        else:
            cat = ppd_cats[0]
    elif cat_count==0:
        return "Androgyny", []
    else:
        cat =  ppd_cats[0]
    if any(x == cat for x in ['dress', 'stockings','bikini']) and gender=='Male':
        return "Androgyny", []
    if cat == 'skirt' and gender == 'Male':
        cat = 'shirt'

    return gender, cat

def title2category(gender, title):
    TITLE= title.upper()
    split1 = re.split(' |-|,', TITLE)
    cats = []
    genderAlert = None
    if any(x in TITLE for x in ['BELT BUCKLE','BELT STRAP']):
        return gender, []
    for s in split1:
        if s in ebay_constants.categories_keywords:
            cats.append(s)
        elif s in ebay_constants.categories_badwords:
            return gender, []
        else:
            pass
        if s in ['WOMAN','WOMAN\'S', 'WOMEN','WOMEN\'S']:
            genderAlert = 'Female'
        if s in ['MAN','MAN\'S', 'MEN','MEN\'S']:
            genderAlert = 'Male'

    gender, ppd_cats = fromCats2ppdCats(gender, cats)
    gender = genderAlert or gender
    return gender, ppd_cats


def ebay_downloader(filename, filesize):
    ftp = ebay_dl_utils.ftp_connection(ebay_dl_utils.us_params)

    start = time()
    sio = StringIO()
    gc.collect()


    def handle_binary(more_data):
        sio.write(more_data)


    try:
        resp = ftp.retrbinary('RETR ' + filename, callback=handle_binary)
    except:
        try:
            ftp = ebay_dl_utils.ftp_connection(ebay_dl_utils.us_params)
            resp = ftp.retrbinary('RETR ' + filename, callback=handle_binary)
        except:
            ftp.quit()
            return

    sio.seek(0)
    zipfile = gzip.GzipFile(fileobj=sio)
    unzipped = zipfile.read()
    # each item is arranged in a dict according to the keys of the first item
    # all items are gathered in a list
    items = csv.DictReader(unzipped.splitlines(), delimiter='\t')
    itemCount = 0
    new_items = 0
    stall = 0
    item = None
    for item in items:
        # verify right category
        mainCategory = item["CATEGORY_NAME"]
        if mainCategory != "Clothing":
            continue
        gender, subCategory = title2category(item["GENDER"], item["OFFER_TITLE"])
        if len(subCategory) < 1:
            continue
        if gender == 'Unisex':
            continue
        # needs to add search for id and etc...
        collection_name = "ebay_" + gender
        if subCategory == "t-shirt":
            # collection_name ="ebay_Tees"
            # exists = db[collection_name].find({'id':item["\xef\xbb\xbfOFFER_ID"]})
            # if exists.count()>1:
            #     db[collection_name].delete_many({'id':item["\xef\xbb\xbfOFFER_ID"]})
            #     exists=[]
            # if exists.count()==0:
            #     generic_dict = ebay2generic(item, gender, subCategory)
            #     db[collection_name].insert_one(generic_dict)
            #     itemCount +=1
            # else:
            #     pass
            continue
        itemCount += 1
        print (itemCount)
        generic_dict = ebay2generic(item, gender, subCategory)
        if generic_dict is None:
            continue
        exists = db[collection_name].find_one({'id': generic_dict['id']})
        if exists and exists["fingerprint"] is not None:
            db[collection_name].update_one({'id': exists['id']}, {"$set": {"download_data.dl_version": today_date,
                                                                           "price": generic_dict["price"]}})
            if exists["status"]["instock"] != generic_dict["status"]["instock"]:
                db[collection_name].update_one({'id': exists['id']}, {"$set": {"status": generic_dict["status"]}})
            elif exists["status"]["instock"] is False and generic_dict["status"]["instock"] is False:
                db[collection_name].update_one({'id': exists['id']}, {"$inc": {"status.days_out": 1}})
            else:
                pass
        else:
            if exists:
                db[collection_name].delete_many({'id': exists['id']})
            else:
                new_items += 1

            while q.count > 250000:
                print("Q full - stolling")
                sleep(600)
                stall += 1

            q.enqueue(generate_mask_and_insert, doc=generic_dict, image_url=generic_dict["images"]["XLarge"],
                      fp_date=today_date, coll=collection_name)
            # db[collection_name].insert_one(generic_dict)

    stop = time()
    db.ebay_download_info.update_one({'type': 'usage'},{"$inc":{'ram_usage':-filesize}})
    if itemCount < 1 and item is not None:
        print("%s = %s is not relevant!" % (filename, item["MERCHANT_NAME"]))
    else:
        if item is None:
            BW = 'black'
        else:
            BW = 'white'
        try:
            db.ebay_download_info.update_one({'type':'store','id': filename[:-7]},{"$set":{
                'dl_duration': stop - start - 600 * stall,
                'items_downloaded': itemCount,
                'B/W' :BW}})
            print(
            "%s (%s) potiential items for %s = %s" % (str(itemCount), str(new_items), item["MERCHANT_NAME"], filename))
        except:
            print ('%s not found in store info' % filename)
    ftp.quit()
    return
