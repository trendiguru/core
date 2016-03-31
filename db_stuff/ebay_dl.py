"""
creation date: 8.3.2016
by: yonti
description: this program downloads all relevant items from ebay through ftp
    - file = store
    - each file is downloaded and then scanned for CATEGORY_NAME= clothing  - this step can be parallelized
    - each item relevant is inserted into our mongo db - using the extended generic dictionary (*see bottom for details)
    - TODO: nlp on description
            run on non english ebay databases
"""
import gc
from ftplib import FTP
from StringIO import StringIO
import gzip
import csv
import time
import datetime
import re
import sys
from .. import constants
from . import ebay_constants
from . import dl_excel
from rq import Queue
from ..fingerprint_core import generate_mask_and_insert
from time import sleep

q = Queue('fingerprint_new', connection=constants.redis_conn)

db = constants.db
status = db.download_status
today_date = str(datetime.datetime.date(datetime.datetime.now()))

def getStoreStatus(store_id,files):
    store_int = int(store_id)
    if store_int in ebay_constants.ebay_blacklist:
        fullname= store_id +".txt.gz"
        files.remove(fullname)
        return "blacklisted"
    elif store_int in ebay_constants.ebay_whitelist:
        return "whitelisted"
    else:
        return "new item"

def getStoreInfo(ftp, files):
    store_info = []
    sio = StringIO()
    def handle_binary(more_data):
        sio.write(more_data)
    resp = ftp.retrbinary('RETR StoreInformation.xml', callback=handle_binary)
    sio.seek(0)
    xml = sio.read()
    split= re.split('</store><store id=',xml)
    split2 = re.split("<store id=|<name><!|></name>|<url><!|></url>",  split[0])
    store_id = split2[1][1:-2]
    status = getStoreStatus(store_id,files)

    item = {'id': store_id,'name': split2[2][7:-2],'link':split2[4][7:-2],
            'dl_duration':0,'items_downloaded':0, 'B/W': 'black', 'modified':"",'status':status}
    store_info.append(item)
    for line in split[1:]:
        split2 = re.split("<name><!|></name>|<url><!|></url>",  line)
        store_id = split2[0][1:-2]
        status = getStoreStatus(store_id,files)
        item = {'id': store_id, 'name': split2[1][7:-2], 'link':split2[3][7:-2],
                'dl_duration':0,'items_downloaded':0, 'B/W': 'black','modified':"",'status':status}
        store_info.append(item)
    files.remove("status.txt")
    files.remove("StoreInformation.xml")
    return store_info


# fills our generic dictionary with the info from ebay
def ebay2generic(item, gender, subcat):
    try:
        generic = {"id": item["\xef\xbb\xbfOFFER_ID"],
                   "categories": subcat,
                   "clickUrl": item["OFFER_URL_MIN_CATEGORY_BID"],
                   "images": item["IMAGE_URL"],
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
    except:
        print item
        generic = item
    return generic

us_params = {"url": "partnersw.ftp.ebaycommercenetwork.com",
          "user": 'p1129643',
          'password': '6F2lqCf4'}


def ftp_connection(params):
    ftp = FTP(params["url"])
    ftp.login(user=params["user"], passwd=params["password"])
    return ftp


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


def theArchiveDoorman():
    for gender in ["Female","Male","Unisex"]:
        collection = db['ebay_'+gender]
        archive = db['ebay_'+gender+'_archive']
        # clean the archive from items older than a week
        archivers = archive.find()
        y_new, m_new, d_new = map(int, today_date.split("-"))
        for item in archivers:
            y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
            days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
            if days_out < 7:
                archive.update_one({'id': item['id']}, {"$set": {"status.days_out": days_out}})
            else:
                archive.delete_one({'id': item['id']})

        # add to the archive items which were not downloaded today but were instock yesterday
        notUpdated = collection.find({"download_data.dl_version": {"$ne": today_date}})
        for item in notUpdated:
            y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
            days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
            if days_out < 7:
                item['status']['instock'] = False
                item['status']['days_out'] = days_out
                archive.insert_one(item)

            collection.delete_one({'id': item['id']})

        # move to the archive all the items which were downloaded today but are out of stock
        outStockers = collection.find({'status.instock': False})
        for item in outStockers:
            archive.insert_one(item)
            collection.delete_one({'id': item['id']})

        archive.reindex()

start_time = time.time()
#connecting to FTP
# username, passwd are for the US - for other countries check the bottom
ftp = ftp_connection(us_params)

# get a list of all files in the directory
data = []
files = []
ftp.dir(data.append)

for line in data:
    elements = line.split()
    filename = elements[8]
    files.append(filename)

store_info = getStoreInfo(ftp,files)

for col in ["Female","Male","Unisex"]:#,"Tees"]:
    col_name = "ebay_"+col
    status_full_path = "collections." + col_name + ".status"
    status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})

for filename in files:
    start = time.time()
    sio = StringIO()
    gc.collect()
    def handle_binary(more_data):
        sio.write(more_data)
    try:
        resp = ftp.retrbinary('RETR '+filename, callback=handle_binary)
    except:
        try:
            ftp = ftp_connection(us_params)
            resp = ftp.retrbinary('RETR '+filename, callback=handle_binary)
        except:
            continue
    sio.seek(0)
    zipfile = gzip.GzipFile(fileobj = sio)
    unzipped = zipfile.read()
    # each item is arranged in a dict according to the keys of the first item
    # all items are gathered in a list
    items = csv.DictReader(unzipped.splitlines(), delimiter='\t')
    itemCount = 0
    new_items = 0
    stoll = 0
    for item in items:
        # verify right category
        mainCategory = item["CATEGORY_NAME"]
        if mainCategory != "Clothing":
            continue
        gender, subCategory = title2category(item["GENDER"], item["OFFER_TITLE"])
        if len(subCategory)<1:
            continue

        #needs to add search for id and etc...
        collection_name = "ebay_"+gender
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
        itemCount +=1
        generic_dict = ebay2generic(item, gender, subCategory)
        exists = db[collection_name].find_one({'id':generic_dict['id']})
        if exists and exists["fingerprint"] is not None:
            db[collection_name].update_one({'id':exists['id']}, {"$set": {"download_data.dl_version":today_date,
                                                                              "price": generic_dict["price"]}})
            if exists["status"]["instock"] != generic_dict["status"]["instock"] :
                db[collection_name].update_one({'id':exists['id']}, {"$set": {"status":generic_dict["status"]}})
            elif exists["status"]["instock"] is False and generic_dict["status"]["instock"] is False:
                db[collection_name].update_one({'id':exists['id']}, {"$inc": {"status.days_out":1}})
            else:
                pass
        else:
            if exists:
                db[collection_name].delete_many({'id':exists['id']})
            else:
                new_items+=1
            while q.count > 250000:
                print( "Q full - stolling")
                sleep(600)
                stoll+=1


            q.enqueue(generate_mask_and_insert, doc=generic_dict, image_url=generic_dict["images"],
                  fp_date=today_date, coll=collection_name)
            # db[collection_name].insert_one(generic_dict)

    stop = time.time()

    if itemCount < 1:
        print("%s = %s is not relevant!" %(filename, item["MERCHANT_NAME"]))
    else:
        try:
            idx = [x["id"] == filename[:-7] for x in store_info].index(True)
            store_info[idx]['dl_duration']=stop- start - 600*stoll
            store_info[idx]['items_downloaded']=itemCount
            store_info[idx]['B/W']= 'white'
            print("%s (%s) potiential items for %s = %s" % (str(itemCount), str(new_items), item["MERCHANT_NAME"],filename))
        except:
            print ('%s not found in store info' %filename)
            continue

ftp.quit()
stop_time = time.time()
total_time = (stop_time-start_time)/3600
for line in data[:-2]:
    s=line.split()
    idx = [x["id"] == s[8][:-7] for x in store_info].index(True)
    store_info[idx]['modified'] = s[5] + " " + s[6] + " at " + s[7]

dl_info = {"date": today_date,
           "dl_duration": total_time,
           "store_info": store_info}

for col in ["Female","Male","Unisex"]:#,"Tees"]:
    col_name = "ebay_"+col
    status_full_path = "collections."+col_name+".status"

    status.update_one({"date": today_date}, {"$set": {status_full_path: "Finishing Up"}})
    # if col != "Tees":
    #     db[col_name].delete_many({'fingerprint': None})

theArchiveDoorman()

dl_excel.mongo2xl('ebay', dl_info)

for col in ["Female","Male","Unisex"]:#,"Tees"]:
    col_name = "ebay_"+col
    status_full_path = "collections."+col_name+".status"
    notes_full_path = "collections."+col_name+".notes"
    new_items = db[col_name].find({'download_data.first_dl': today_date}).count()
    status.update_one({"date": today_date}, {"$set": {status_full_path: "Done",
                                                      notes_full_path: new_items}})

print("ebay Download is Done")
sys.exit(0)


'''
ftp codes per country:
can be found in our google drive

extended generic dictionary -
    has all the categories from the generic db + extra key which contains all the raw info from ebay
ebay dictionary:

    OFFER_ID: should be unique - ends with ==
    OFFER_TITLE: short description
    PRICE: in dollars
    OFFER_DESCRIPTION: long description
    MERCHANT_ID: the same as the file name
    MERCHANT_NAME: optional
    MERCHANT_SKU_NUMBER: optional
    SDC_PRODUCT_ID: optional
    ISBN: optional
    UPC: optional
    MPN: optional
    CATEGORY_NAME: we are looking only for 'Clothing'
    STOCK: status - if out add number of days out
    CATEGORY_ID: 31515 is the category id for clothing
    STOCK_DESCRIPTION: extra comments like 'Free Shipping on orders over $100'
    IMAGE_URL: can have more than one img - find the best resolution
    MAX_IMG_WIDTH: 500
    MAX_IMG_HEIGHT: 575
    SHIPPING_RATE: optional
    SHIPPING_WEIGHT: 0.0
    ZIP_CODE: optional
    MANUFACTURER: Karen Kane
    ATTRIBUTES: example 'Condition=New|Gender and Age=Women|Apparel Type=Dresses|Color=Black' - we need to use this info
    ARTIST_AUTHOR: optional
    MIN_CATEGORY_BID: 35
    MERCHANT_CATEGORY_BID: 35
    OFFER_URL_MIN_CATEGORY_BID: buy url
        http://rover.ebay.com/rover/13/0/19/DealFrame/DealFrame.cmp?bm=736&BEFID=31515&acode=742&code=742&aon=&crawler_id=522709&dealId=-4VvBvDJO29eqtnUJQtxmw%3D%3D&searchID=&url=http%3A%2F%2Ftracking.searchmarketing.com%2Fclick.asp%3Faid%3D730009810000108420%26sdc_id%3D%7Bsdc_id%7D&DealName=Karen%20Kane%20Contrast%20Dot%20Scuba%20Dress%20Black-w-Off-White&MerchantID=522709&HasLink=yes&category=0&AR=-1&NG=1&GR=1&ND=1&PN=1&RR=-1&ST=&MN=msnFeed&FPT=SDCF&NDS=1&NMS=1&NDP=1&MRS=&PD=0&brnId=2455&lnkId=8097282&Issdt=160224010101&IsFtr=0&IsSmart=0&dlprc=89.4&SKU=KK-4L19557
    OFFER_URL_MERCHANT_CATEGORY_BID: buy url - same for all but bm - what it means?
        http://rover.ebay.com/rover/13/0/19/DealFrame/DealFrame.cmp?bm=722&BEFID=31515&acode=742&code=742&aon=&crawler_id=522709&dealId=-4VvBvDJO29eqtnUJQtxmw%3D%3D&searchID=&url=http%3A%2F%2Ftracking.searchmarketing.com%2Fclick.asp%3Faid%3D730009810000108420%26sdc_id%3D%7Bsdc_id%7D&DealName=Karen%20Kane%20Contrast%20Dot%20Scuba%20Dress%20Black-w-Off-White&MerchantID=522709&HasLink=yes&category=0&AR=-1&NG=1&GR=1&ND=1&PN=1&RR=-1&ST=&MN=msnFeed&FPT=SDCF&NDS=1&NMS=1&NDP=1&MRS=&PD=0&brnId=2455&lnkId=8097282&Issdt=160224010101&IsFtr=0&IsSmart=0&dlprc=89.4&SKU=KK-4L19557
    PRODUCT_MPN: optional
    PRODUCT_UPC: optional
    PRODUCT_NAME: optional
    ORIGINAL_PRICE: 149.00
    MOBILE_URL: optional
    PRODUCT_BULLET_POINT_1_5: optional
    ALT_IMAGE_URL_1_5: optional
    UNIT_PRICE: optional
    PRODUCT_LAUNCH_DATE: optional
    CONDITION: NEW - might be also found in the attributes
    ESTIMATED_SHIP_DATE: optional
    COUPON_CODE:optional
    COUPON_CODE_DESCRIPTION: optional
    BUNDLE: 0
    MERCHANDISING_TYPE: optional
    DEAL_SCORE: optional
    ORIG_OFFER_URL_NOTRACKING: use this url to reach item without using our id
                                i.e, http://tracking.searchmarketing.com/click.asp?aid=730009810000108420&sdc_id={sdc_id}
    PARENT_SKU: optional
    PARENT_NAME: optional
    GENDER: Female  - important
    SIZE: optional
    COLOR: Black - its not accurate when there is more than one color
    AGE_GROUP: Adult
    TOP_CATEGORY_NAME: Clothing and Accessories - it should be the same for all our product
    TOP_CATEGORY_ID: 2010 - it should be the same for all our product
    ORIG_MERCHANT_CATEGORY_NAME: Dress - need to check what the main categories
    EBAY_PRODUCT_ID: optional
    ENERGY_EFFICIENCY_CLASS: optional
    QUANTITY: 0
    UPDATE_TIME: 1456186249968 - maybe we can use this
    RPP_EVENT_ID: optional
    RPP_EVENT_NAME: optional
    RPP_EVENT_IMAGE_URL: optional
    FLEX_FIELD: optional
'''