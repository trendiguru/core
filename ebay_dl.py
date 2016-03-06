'''
creation date: 21.2.2016
by: yonti
description: this program downloads all relevant items from ebay through ftp
    - file = store
    - each file is downloaded and then scanned for CATEGORY_NAME= clothing  - this step can be parallelized
    - each item relevant is inserted into our mongo db - using the extended generic dictionary (*see bottom for details)
    - TODO: nlp on description
            get relevant attributes from ATTRIBUTES key
            run on non english ebay databases

'''

from ftplib import FTP
from StringIO import StringIO
import gzip
import csv
import time
import datetime
import re
from . import constants
from . import ebay_constants
db = constants.db
db.ebay_Female.delete_many({})
# db.ebay_Male.delete_many({})
# db.ebay_Unisex.delete_many({})
today_date = str(datetime.datetime.date(datetime.datetime.now()))

ebaysNotRelevant = ebay_constants.blacklist_stores
# b = open("/home/developer/python-packages/trendi/Yonti/ebay_blacklist.txt",'r')
# for store in b:
#     if store not in ebaysNotRelevant:
#         ebaysNotRelevant.append(store)

# manual check [20478,
# fills our generic dictionary with the info from ebay
def ebay2generic(item, subcats):
    try:
        generic = {"id": item["\xef\xbb\xbfOFFER_ID"],
                   "categories": subcats,
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
                   "gender": item["GENDER"],
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

def title2category(title):
    TITLE= title.upper()
    split1 = re.split(' |-', TITLE)
    cats = []
    for s in split1:
        if s in ebay_constants.categories_keywords:
            cats.append(s)
        elif s in ebay_constants.blacklist_categories:
            return []
        else:
            pass
    return cats

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
        gender = item["GENDER"]
        if mainCategory != "Clothing" or gender !='Female':
            continue
        subCategorys = title2category(item["OFFER_TITLE"])
        if len(subCategorys)<1:
            continue
        itemCount +=1
        #needs to add search for id and etc...
        generic_dict = ebay2generic(item, subCategorys)
        if gender == "Female":
            db.ebay_Female.insert_one(generic_dict)
        elif gender == "Male":
            db.ebay_Male.insert_one(generic_dict)
        else:
            db.ebay_Unisex.insert_one(generic_dict)
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
print ("download_time = %s" % str(total_time))

print ("\n\ncategories:")
for cat in categories:
    print(cat)

print ("\n\nblacklist:")
for black in black_list:
    print(black)

print ("\n\nwhitelist:")
for w in white_list:
    print(w)

'''
ftp codes per country:
country  |Account               |FTP address                            |Username   |Password   |TrackingID
US       |Trendi Guru-1129643   |partnersw.ftp.ebaycommercenetwork.com  |p1129643   |6F2lqCf4   |8097282
DE       |Trendi Guru-1036365   |partnersw.ftp.ebaycommercenetwork.com  |p1036365   |vmo7tzXb   |8097281
UK       |Trendi Guru-1129924   |partnersw.ftp.ebaycommercenetwork.com  |p1129924   |J5Swk69V   |8097283
FR       |Trendi Guru-1129925   |partnersw.ftp.ebaycommercenetwork.com  |p1129925   |frRf0CR6   |8097284
AU       |Trendi Guru-1129927   |partnersw.ftp.ebaycommercenetwork.com  |p1129927   |4awYGf27   |8097285

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