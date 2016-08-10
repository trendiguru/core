"""
playground for testing the ebay API
"""

import collections
import datetime
import hashlib
from time import time

import requests
import xmltodict

from core.db_stuff.archive.ebay_global_constants import APPID, ebay_male_relevant_categories, \
    ebay_female_relevant_categories,ebay_not_relevant_categories, MAIN_CATEGORY,VERSION, \
    ebay_paperdoll_men,ebay_paperdoll_women
from core import constants

db = constants.db
today_date = str(datetime.datetime.date(datetime.datetime.now()))


def GET(CAT_ID):
    res = requests.get("http://open.api.ebay.com/Shopping?"
                       "callname=GetCategoryInfo"
                       "&appid=" + APPID + \
                       "&version=" + VERSION + \
                       "&CategoryID=" + CAT_ID + \
                       "&IncludeSelector=ChildCategories")
    category_dict = xmltodict.parse(res.text)
    topCategories = []
    for cat in category_dict['GetCategoryInfoResponse']['CategoryArray']['Category']:
        tmp = {'CategoryName': cat['CategoryName'],
               'CategoryId': cat['CategoryID'],
               'LeafCategory': cat['LeafCategory']}
        topCategories.append (tmp )
    return topCategories

def breakItDown(categories_list):
    for x in categories_list:
        return x['CategoryName'], x['CategoryId']

def getTopCategories():
    # find which categories exists in clothing and shoes (11450)
    topCategories = GET(MAIN_CATEGORY)

    top = []
    for cat in topCategories:
        if any([x for x in ['Men','Women'] if x in cat['CategoryName']]):
            cat_list = GET(cat['CategoryId'])
            children = []
            for i,x in enumerate(cat_list):
                if i==0:
                    continue
                tmp= {'name': x['CategoryName'], 'idx':x['CategoryId']}
                children.append(tmp)
            tmp_top = {'name':cat['CategoryName'],'idx':cat['CategoryId'], 'children_count':len(children),'children':children}
            top.append(tmp_top)

    return top

def getBrandHistogram(idx):
    res = requests.get("http://svcs.ebay.com/services/search/FindingService/v1?"
                       "OPERATION-NAME=findItemsAdvanced&SECURITY-APPNAME="+APPID+ \
                       "&RESPONSE-DATA-FORMAT=XML&REST-PAYLOAD&outputSelector=AspectHistogram"
                       "&categoryId=" + idx + \
                       "&paginationInput.entriesPerPage=1&AvailableItemsOnly=true"
                       "&itemFilter(0).name=ListingType&itemFilter(0).value=FixedPrice"
                       "&itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true"
                       "&itemFilter(2).name=Condition&itemFilter(2).value=New")
    category_dict = xmltodict.parse(res.text)
    brands = []
    if 'aspectHistogramContainer' not in category_dict['findItemsAdvancedResponse'].keys():
        print('break')
        return brands
    for x in category_dict['findItemsAdvancedResponse']['aspectHistogramContainer']['aspect']:
        if type(x) != collections.OrderedDict:
            print('continue')
            continue
        if x['@name']=='Brand':
            for y in x['valueHistogram']:
                # print(y)
                tmp = {'brand':y['@valueName'],
                       'count':y['count']}
                brands.append(tmp)

    return brands


def fill_brand_info(dictlist):
    dictwithbrands = []
    for topCat in dictlist:
        tmp = {'name':topCat['name'],'idx':topCat['idx'], 'children':[]}
        for child in topCat['children']:
            name = child['name']
            idx = child['idx']
            # print(name)
            if name in ebay_not_relevant_categories:
                continue
            brands_info = getBrandHistogram(idx)
            tmp_child = {'idx':idx, 'childName':name, 'brands':brands_info}
            tmp['children'].append(tmp_child)
        dictwithbrands.append(tmp)
    return dictwithbrands


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def getItemsbyBrand(category,category_idx,brand,gender,collection,pageNumber):
    if pageNumber> 100:
        return
    res = requests.get("http://svcs.ebay.com/services/search/FindingService/v1?"
                       "OPERATION-NAME=findItemsAdvanced&SERVICE-VERSION=1.12.0"
                       "&SECURITY-APPNAME="+APPID+ \
                       "&RESPONSE-DATA-FORMAT=XML&REST-PAYLOAD"
                       "&categoryId="+ category_idx+ \
                       "&paginationInput.entriesPerPage=100"
                       "&AvailableItemsOnly=true"
                       "&itemFilter(0).name=ListingType&itemFilter(0).value=FixedPrice"
                       "&itemFilter(1).name=HideDuplicateItems&itemFilter(1).value=true"
                       "&itemFilter(2).name=Condition&itemFilter(2).value=New"
                       "&paginationInput.pageNumber="+str(pageNumber)+ \
                       "&aspectFilter.aspectName=Brand&aspectFilter.aspectValueName="+brand)

    if res.status_code != 200:
        print ('bad response')
        return

    response = xmltodict.parse(res.text)

    items = response['findItemsAdvancedResponse']['searchResult']

    p = response['findItemsAdvancedResponse']['paginationOutput']
    pagination = {'pageNumber': int(p['pageNumber']),
                  'totalPages': int(p['totalPages']),
                  'totalEntries': int(p['totalEntries'])}

    if pagination['totalEntries']<2:
        return

    for item in items['item']:
        try:
            itemId = item['itemId']
            exists = collection.find_one({'id':itemId})
            if exists:
                print ('item already exists')
                continue

            price = {'price':item['sellingStatus']['currentPrice']['#text'],
                     'currency': item['sellingStatus']['currentPrice']['@currencyId']}
            s = item['sellingStatus']['sellingState']
            if s == 'Active':
                status = 'instock'
            else:
                status = 'outOfStock'

            img_url = item['galleryURL']

            generic = {"id": [item['itemId']],
                       "categories":category,
                       "clickUrl": item['viewItemURL'],
                       "images": {"XLarge": img_url},
                       "status": status,
                       "shortDescription": item['title'],
                       "longDescription": [],
                       "price": price,
                       "brand": brand,
                       "download_data": {'dl_version': today_date,
                                         'first_dl': today_date,
                                         'fp_version': constants.fingerprint_version},
                       "fingerprint": None,
                       "gender": gender,
                       "shippingInfo": item['shippingInfo'],
                       "ebay_raw": item}

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
        except:
            print ('something wrong with the item dict')


    p =  response['findItemsAdvancedResponse']['paginationOutput']
    pagination =  { 'pageNumber': int(p['pageNumber']),
                    'totalPages': int(p['totalPages']),
                    'totalEntries': int(p['totalEntries'])}

    if pagination['pageNumber'] < pagination['totalPages']:
        return getItemsbyBrand(category,category_idx,brand,gender,collection,pagination['pageNumber'] + 1)
    else:
        return


def download(allCatsandBrands):
    for TopCat in allCatsandBrands:
        print (TopCat['name'])
        if 'Women' in TopCat['name']:
            gender = 'Female'
            relevant = ebay_female_relevant_categories
            category_convertor = ebay_paperdoll_women
            col = db.ebay_global_Female
        else:
            gender = 'Male'
            relevant = ebay_male_relevant_categories
            category_convertor = ebay_paperdoll_men
            col = db.ebay_global_Male

        for cat in TopCat['children']:
            if cat['childName'] in relevant:
                idx = cat['idx']
                c = cat['childName']
                print(c)
                category = category_convertor[c]
                #tmp
                catexists = col.find_one({'categories':category})
                if catexists:
                    continue
                for brand in cat['brands']:
                    startT= time()
                    print(brand)
                    if brand['count']=='0':
                        print ('no items')
                        continue
                    getItemsbyBrand(category,idx, brand['brand'], gender,col, 1)
                    stopT = time()
                    print (stopT-startT)

if __name__=='__main__':
    # db.ebay_global_Female.delete_many({})
    # db.ebay_global_Male.delete_many({})
    start = time()
    top_categories = getTopCategories()
    stage1 = time()
    print ('getTopCategories duration = %s' %(str(stage1-start)))
    brand_info = fill_brand_info(top_categories)
    stage2 = time()
    print ('fiil_brand_info duration = %s' % (str(stage2 - stage1)))
    download(brand_info)
    stage3 = time()
    print ('download duration = %s' % (str(stage3 - stage1)))

    print ('done!')
else:
    start = time()
    top_categories = getTopCategories()
    stage1 = time()
    print ('getTopCategories duration = %s' % (str(stage1 - start)))
    brand_info = fill_brand_info(top_categories)
    stage2 = time()
    print ('fiil_brand_info duration = %s' % (str(stage2 - stage1)))
    download(brand_info)
    stage3 = time()
    print ('download duration = %s' % (str(stage3 - stage1)))
    print ('done!')

# for x in brand_info:
#     for y in x['children']:
#         print y['brands']
#         raw_input()

"""
1. get all categories - make list of dict - cat + id
2. for each category - get all brand names and count
3. for each brand - retrieve results
4. use all info
** there are items with no brand / homemade

[u'@xmlns', u'ack', u'version', u'timestamp', u'searchResult', u'paginationOutput', u'itemSearchURL']

"""
