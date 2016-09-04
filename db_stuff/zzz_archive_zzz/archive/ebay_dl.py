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
import datetime
import re
from StringIO import StringIO
from time import sleep,time

from rq import Queue

from core import constants
from core.db_stuff.annoy.fanni import plantForests4AllCategories
from core.db_stuff.ebay import ebay_constants
from core.db_stuff.general import dl_excel
from core.db_stuff.zzz_archive_zzz.archive import ebay_dl_utils
from .ebay_dl_worker import ebay_downloader

q = Queue('ebay_worker', connection=constants.redis_conn)
forest = Queue('annoy_forest', connection=constants.redis_conn)
db = constants.db
status = db.download_status
today_date = str(datetime.datetime.date(datetime.datetime.now()))
yesterday = str(datetime.datetime.date(datetime.datetime.now() - datetime.timedelta(days=1)))


def getStoreStatus(store_id,files):
    store_int = int(store_id)
    fullname = store_id + ".txt.gz"
    last_modified = filter(lambda store: store['name'] == fullname, files)
    if len(last_modified)== 0 :
        last_modified = 'check for error'
    else:
        last_modified=last_modified[0]['last_modified']
    if store_int in ebay_constants.ebay_blacklist:
        files = filter(lambda x:x.get('name')!=fullname,files)
        return last_modified, files, "blacklisted"
    elif store_int in ebay_constants.ebay_whitelist:
        return last_modified, files, "whitelisted"
    else:
        return last_modified, files, "new item"


def StoreInfo(ftp, files):
    sio = StringIO()
    def handle_binary(more_data):
        sio.write(more_data)
    resp = ftp.retrbinary('RETR StoreInformation.xml', callback=handle_binary)
    sio.seek(0)
    xml = sio.read()
    split= re.split('</store><store id=',xml)
    split2 = re.split("<store id=|<name><!|></name>|<url><!|></url>",  split[0])
    store_id = split2[1][1:-2]
    last_modified ,files, status = getStoreStatus(store_id,files)
    item = {'type':'store','id': store_id,'name': split2[2][7:-2],'link':split2[4][7:-2],
            'dl_duration':0,'items_downloaded':0, 'B/W': 'black','status':status,
            'modified': last_modified}
    db.ebay_download_info.insert_one(item)
    for line in split[1:]:
        split2 = re.split("<name><!|></name>|<url><!|></url>",  line)
        store_id = split2[0][1:-2]
        last_modified, files, status = getStoreStatus(store_id,files)
        item = {'type':'store','id': store_id, 'name': split2[1][7:-2], 'link':split2[3][7:-2],
                'dl_duration':0,'items_downloaded':0, 'B/W': 'black','status':status,
                'modified': last_modified}
        db.ebay_download_info.insert_one(item)
    # files = filter(lambda x: x.get('name') == "status.txt",files)
    # files = filter(lambda x: x.get('name') == "StoreInformation.xml",files)
    return files


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

        # add to the archive items which were not downloaded today or yesterday but were instock yesterday
        notUpdated = collection.find({"download_data.dl_version": {"$nin": [today_date,yesterday]}})
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


db.ebay_download_info.delete_many({})
start_time = time()

#connecting to FTP
ftp = ebay_dl_utils.ftp_connection(ebay_dl_utils.us_params)

# get a list of all files in the directory
data = []
files = []
ftp.dir(data.append)

for line in data[:-2]:
    elements = line.split()
    filename = elements[8]
    filesize = elements[4]
    modified = elements[5] + " " + elements[6] + " at " + elements[7]
    files.append({'name':filename, 'size':filesize,'last_modified': modified})

files = StoreInfo(ftp,files)
ftp.quit()

for col in ["Female","Male","Unisex"]:#,"Tees"]:
    col_name = "ebay_"+col
    status_full_path = "collections." + col_name + ".status"
    status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})

print('total number of stores to download = %s' %(len(files)))

for x,file in enumerate(files):
    filename = file['name']
    filesize = int(file['size'])
    while q.count>2:
        sleep(300)
    print ('enqueued store id = %s' %(filename) )
    q.enqueue(ebay_downloader, args=(filename, filesize), timeout=5400)

#wait for workers
while q.count>0:
    sleep(1000)
    print("waiting for workers to finish")

stop_time = time()
total_time = (stop_time-start_time)/3600
store_info = db.ebay_download_info.find({'type':'store'})

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
    forest_job = forest.enqueue(plantForests4AllCategories, col_name=col_name, timeout=3600)
    while not forest_job.is_finished and not forest_job.is_failed:
        sleep(300)
    if forest_job.is_failed:
        print ('annoy plant forest failed')


print("ebay Download is Done")
'''
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