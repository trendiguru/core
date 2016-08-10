import csv
import datetime
import gc
import gzip
import hashlib
import re
import sys
from StringIO import StringIO
from time import sleep,time

import psutil
from rq import Queue

from core import constants, Utils
from core.db_stuff import ebay_constants
from core.db_stuff.archive import ebay_dl_utils
from core.fingerprint_core import generate_mask_and_insert

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


def ebay2generic(item, info):
    try:
        full_img_url = item["IMAGE_URL"]
        generic = {"id": [info["id"]],
                   "categories": info["categories"],
                   "clickUrl": item["OFFER_URL_MIN_CATEGORY_BID"],
                   "images": {"XLarge": full_img_url},
                   "status": info["status"],
                   "shortDescription": item["OFFER_TITLE"],
                   "longDescription": item["OFFER_DESCRIPTION"],
                   "price":  info["price"],
                   "Brand" : item["MANUFACTURER"],
                   "Site" : item["MERCHANT_NAME"],
                   "download_data": {'dl_version': today_date,
                                     'first_dl': today_date,
                                     'fp_version': constants.fingerprint_version},
                   "fingerprint": None,
                   "gender": info["gender"],
                   "ebay_raw": item}

        image = Utils.get_cv2_img_array(full_img_url)
        if image is None:
            #try again
            if 'https://' in full_img_url:
                image = Utils.get_cv2_img_array(full_img_url[8:])
            elif 'http://' in full_img_url:
                image = Utils.get_cv2_img_array(full_img_url[7:])
            else:
                image,generic = None, None

            if image is None:
                generic = None
                return image,generic

        img_hash = get_hash(image)
        generic["img_hash"] = img_hash

    except:
        print item
        generic = None
        image = None
    return image, generic


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
    if any(x == cat for x in ['dress', 'stockings', 'bikini']) and gender == 'Unisex':
        return "Female", 'dress'
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

def getImportantInfoOnly(item):
    gender, subCategory = title2category(item["GENDER"], item["OFFER_TITLE"])
    item_id = item["\xef\xbb\xbfOFFER_ID"]
    price = item["PRICE"]
    status = item["STOCK"]
    info = {"id": item_id,
            "gender": gender,
            "categories": subCategory,
            "price": {'price': price,
                      'priceLabel': "USD"},
            "status": {"instock": True, "days_out": 0}}
    if status != "In Stock":
        info["status"]["instock"] = False
    return info


def startORstall(filesize):
    total_ram = int(psutil.virtual_memory()[0])
    available_ram = int(psutil.virtual_memory()[1])
    # if filesize > 0.5*total_ram:
    #     if filesize < 0.7*available_ram:
    #         return True
    #     else:
    #         return False
    # else:
    #     if filesize < 0.5*available_ram:
    #         return True
    #     else:
    #         return False
    if filesize>(0.75*available_ram):
        return False
    return True


def ebay_downloader(filename, filesize):
    if not startORstall(filesize):
        q.enqueue(ebay_downloader, args=(filename, filesize), timeout=5400)
        sleep(150)
        return

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
        minimal_info = getImportantInfoOnly(item)
        if len(minimal_info["categories"]) < 1:
            continue
        # needs to add search for id and etc...
        collection_name = "ebay_" + minimal_info["gender"]
        if minimal_info["categories"] == "t-shirt":
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
        exists = db[collection_name].find_one({'id': {'$in':[minimal_info['id']]}})
        existsPlusHash = db[collection_name].find_one({'id': {'$in':[minimal_info['id']]}, "img_hash":{"$exists":1}})
        if exists and existsPlusHash and exists["fingerprint"] is not None:
            if not exists['download_data']['dl_version']== today_date:
                db[collection_name].update_one({'_id': exists['_id']}, {"$set": {"download_data.dl_version": today_date,
                                                                           "price": minimal_info["price"]}})
                if exists["status"]["instock"] != minimal_info["status"]["instock"]:
                    db[collection_name].update_one({'_id': exists['_id']}, {"$set": {"status": minimal_info["status"]}})
                elif exists["status"]["instock"] is False and minimal_info["status"]["instock"] is False:
                    db[collection_name].update_one({'_id': exists['_id']}, {"$inc": {"status.days_out": 1}})
                else:
                    pass
            else:
                pass
        else:
            if exists and existsPlusHash: #but fingerprint is none!
                db[collection_name].delete_many({'_id': exists['_id']})
            elif exists and not existsPlusHash: #got no hash!
                image = Utils.get_cv2_img_array(item["IMAGE_URL"])
                if image is None:
                    db[collection_name].delete_many({'id': exists['id']})
                else:
                    img_hash = get_hash(image)
                    db[collection_name].update_one({'_id': exists['_id']}, {"$set": {"img_hash": img_hash}})
                    continue
            else:
            #check if in archive and has hash
                archive = collection_name+"_archive"
                existsInArchive = db[archive].find_one({'id': {'$in':[minimal_info['id']]},"img_hash":{"$exists":1}})
                if existsInArchive and existsInArchive["fingerprint"] is not None:
                    existsInArchive["download_data"]["dl_version"]=today_date
                    existsInArchive["price"] = minimal_info["price"]
                    if minimal_info["status"]["instock"]:
                        existsInArchive["status"] = minimal_info["status"]
                        db[archive].delete_one({'_id': existsInArchive['_id']})
                        db[collection_name].insert_one(existsInArchive)
                    else:
                        db[archive].update_one({"_id":existsInArchive['_id']},{"$set":{"download_data.dl_version": today_date}})
                    continue
                elif existsInArchive:
                    db[archive].delete_one({"_id": existsInArchive['_id']})
                else:
                    pass

                while q.count > 250000:
                    print("Q full - stolling")
                    sleep(600)
                    stall += 1

                img, generic_dict = ebay2generic(item, minimal_info)
                if generic_dict is None or img is None:
                    print ('img download failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    continue
                #check if hash already exists:
                hashexists  = db[collection_name].find_one({'img_hash': generic_dict['img_hash']})
                hashexistsInArchive = db[archive].find_one({'img_hash': generic_dict['img_hash']})
                if hashexists:
                    id_list = hashexists['id'] + generic_dict['id']
                    db[collection_name].update_one({'_id': hashexists['_id']},{'$set':{'id':id_list}})
                    print ('hash exists')
                elif hashexistsInArchive :
                    id_list = hashexistsInArchive['id'] + generic_dict['id']
                    hashexistsInArchive = db[archive].find_one_and_update({'_id': hashexistsInArchive['_id']}, {'$set': {'id': id_list}})
                    if minimal_info["status"]["instock"]:
                        print ('hash exists in archive and is now instock')
                        hashexistsInArchive["status"] = minimal_info["status"]
                        db[archive].delete_one({'_id': hashexistsInArchive['_id']})
                        db[collection_name].insert_one(hashexistsInArchive)
                    else:
                        print('hash exists in archive but out of stock')

                else:
                    new_items+=1
                    print('new item')
                    q.enqueue(generate_mask_and_insert, doc=generic_dict, image_url=generic_dict["images"]["XLarge"],
                          fp_date=today_date, coll=collection_name, img=img)

    print(' new items = %d' %(new_items))
    stop = time()

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
