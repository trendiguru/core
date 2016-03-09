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

from ftplib import FTP
from StringIO import StringIO
import gzip
import csv
import time
import datetime
import re
from . import constants
from . import ebay_constants
from . import dl_excel
db = constants.db
db.ebay_Female.delete_many({})
db.ebay_Male.delete_many({})
db.ebay_Unisex.delete_many({})
today_date = str(datetime.datetime.date(datetime.datetime.now()))

ebaysNotRelevant = ebay_constants.ebay_blacklist

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
        elif 'blazer' in ppd_cats:
            cat = 'blazer'
        elif 'bikini' in ppd_cats:
            return 'bikini'
        elif 'swimsuit' in ppd_cats:
            cat =  'swimsuit'
        elif 'sweater' in ppd_cats:
            return 'sweater'
        elif 'sweatshirt' in ppd_cats:
            cat =  'sweatshirt'
        elif all(x in ppd_cats for x in ['dress','shirt']):
            cat =  'shirt'
        elif all(x in ppd_cats for x in ['dress','pants']):
            cat =  'pants'
        else:
            cat = ppd_cats[0]
    elif cat_count==0:
        return "Androgyny",[]
    else:
        cat =  ppd_cats[0]
    if any(x == cat for x in ['dress', 'skirt']):
        gender = 'Female'
    return gender, cat


def title2category(gender, title):
    TITLE= title.upper()
    split1 = re.split(' |-', TITLE)
    cats = []
    for s in split1:
        if s in ebay_constants.categories_keywords:
            cats.append(s)
        elif s in ebay_constants.categories_badwords:
            return gender, []
        else:
            pass
    gender, ppd_cats = fromCats2ppdCats(gender, cats)
    return gender, ppd_cats

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


#remove not relevant stores/files
for store in ebaysNotRelevant:
    fullname= str(store) +".txt.gz"
    if fullname in files:
        files.remove(fullname)

files.remove("status.txt")
files.remove("StoreInformation.xml")

# temporary
categories =[]
black_list = []
white_list = []

for filename in files:
    start = time.time()

    sio = StringIO()
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
    for item in items:
        # verify right category
        mainCategory = item["CATEGORY_NAME"]
        if mainCategory != "Clothing":
            continue
        gender, subCategory = title2category(item["GENDER"], item["OFFER_TITLE"])
        if len(subCategory)<1:
            continue
        itemCount +=1
        #needs to add search for id and etc...
        collection_name = "ebay_"+gender
        #check if exists
        #to do

        generic_dict = ebay2generic(item, gender, subCategory)
        db[collection_name].insert_one(generic_dict)

    stop = time.time()
    if itemCount < 10:
        black_list.append(filename)
        print("%s = %s is not relevant!" %(filename, item["MERCHANT_NAME"]))
    else:
        white_list.append(filename)
        print("%s potiential items for %s = %s" % (str(itemCount), item["MERCHANT_NAME"],filename))
    print "item download+scraping took %s secs" % str(stop-start)

ftp.quit()
stop_time = time.time()
total_time = (stop_time-start_time)/60
raw_data =[]
for line in data:
    s=line.split()
    status = "whitelist" if s[8] in white_list else "blacklist"
    sorted_data = [s[8], s[5]+ " " + s[6] + " at " + s[7], s[4], status]
    raw_data.append(sorted_data)

white_list_new = [x for x in white_list if x not in ebay_constants.ebay_whitelist]
dl_info = {"date": today_date,
           "dl_duration": total_time,
           "blacklist" : black_list,
           "whitelist" : white_list_new,
           "raw_data": raw_data}

dl_excel.mongo2xl('ebay', dl_info)

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