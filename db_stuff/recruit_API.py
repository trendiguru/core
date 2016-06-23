"""
playground for testing the recruit API
"""
import requests
import json
from datetime import datetime
from recruit_constants import api_stock, recruitID2generalCategory
from ..constants import db, fingerprint_version
today_date = str(datetime.date(datetime.now()))

def generate_genreid(gender, main_category, sub_category):
    if gender is 'Female':
        gen = '01'
    else:
        gen = '02'
    if main_category>10:
        m = str(main_category)
    else:
        m = '0'+str(main_category)
    if sub_category > 10:
        s = str(sub_category)
    else:
        s = '0' + str(sub_category)
    genreid = gen+m+s+'0000'
    return genreid


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


def API4printing(genreId, gender, category_name, skip):
    success, dic = GET_ByGenreId(genreId, instock=False)
    if not success:
        skip += 1
        return skip
    allitems = dic['count']
    status, dic = GET_ByGenreId(genreId, instock=True)
    top_cat = dic["itemInfoList"][0]["genreInfoList"][0]['genreName']
    try:
        japanese_name = dic["itemInfoList"][0]["genreInfoList"][2]['genreName']
        sec_cat = dic["itemInfoList"][0]["genreInfoList"][1]['genreName']
    except:
        try:
            sec_cat = japanese_name = dic["itemInfoList"][0]["genreInfoList"][1]['genreName']
        except:
            sec_cat = japanese_name = dic["itemInfoList"][0]["genreInfoList"][0]['genreName']
    instock_only = dic['count']

    print( 'gender: %s, genreId: %s, category_name: %s , total_count: %s, instock: %s, , japanese: %s , %s , %s'
        % (gender, genreId, category_name, allitems, instock_only, japanese_name, sec_cat, top_cat))
    return skip


def getCategoryName(genreId):
    tmp = [x for x in api_stock if x['genreId']==genreId]
    if len(tmp)>0:
        category_name = tmp[0]['category_name']
    else:
        category_name = 'new_cat'
    return category_name


def printCategories(only_known=True):
    if only_known:
        for cat in api_stock:
            genreId = cat['genreId']
            category_name = cat['category_name']
            gender = cat['gender']
            API4printing(genreId, gender, category_name, 0)
        print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxox')
    else:
        for gender in ['Female', 'Male']:
            for m in range(99):
                skip = 0
                for s in range(99):
                    genreId = generate_genreid(gender, m, s)
                    category_name = getCategoryName(genreId)
                    skip = API4printing(genreId, gender, category_name, skip)
                    if skip == 3:
                        break
                print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxox')

"""
use printCategories to scan the api and print all categories under ladies' and men's fashion
#printCategories(False)
"""

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


def download_recruit():
    db.recruit_Female.delete_many({})
    db.recruit_Male.delete_many({})

    for genreId in recruitID2generalCategory.keys():
        success, response_dict = GET_ByGenreId(genreId, limit=100, instock=True)
        if not success:
            continue
        if genreId[1]=='1':
            gender = 'Female'
        else:
            gender='Male'
        new_items = total_items = 0
        category = recruitID2generalCategory[genreId]
        new_inserts, total = process_items(response_dict["itemInfoList"], gender, category)
        new_items += new_inserts
        total_items += total
        pageCount = int(response_dict['pageCount'])
        if pageCount>999:
            pageCount=999
        for i in range(2,pageCount+1):
            success, response_dict = GET_ByGenreId(genreId,page=i, limit=100, instock=True)
            if not success:
                continue
            new_inserts, total = process_items(response_dict["itemInfoList"], gender, category)
            new_items += new_inserts
            total_items += total

        print ('genreId: %s, categoryName: %s, total: %s, new: %s'
               %(genreId, category, str(total_items), str(new_items)))

if __name__=='__main__':
    download_recruit()


# GET_gnereid(generate_genreid('Female',0,0))
# if __name__=='__main__':
#     # db.ebay_global_Female.delete_many({})
#     # db.ebay_global_Male.delete_many({})
#     start = time()
#     top_categories = getTopCategories()
#     stage1 = time()
#     print ('getTopCategories duration = %s' %(str(stage1-start)))
#     brand_info = fill_brand_info(top_categories)
#     stage2 = time()
#     print ('fiil_brand_info duration = %s' % (str(stage2 - stage1)))
#     download(brand_info)
#     stage3 = time()
#     print ('download duration = %s' % (str(stage3 - stage1)))
#
#     print ('done!')
# else:
#     start = time()
#     top_categories = getTopCategories()
#     stage1 = time()
#     print ('getTopCategories duration = %s' % (str(stage1 - start)))
#     brand_info = fill_brand_info(top_categories)
#     stage2 = time()
#     print ('fiil_brand_info duration = %s' % (str(stage2 - stage1)))
#     download(brand_info)
#     stage3 = time()
#     print ('download duration = %s' % (str(stage3 - stage1)))
#     print ('done!')

# for x in brand_info:
#     for y in x['children']:
#         print y['brands']
#         raw_input()

"""
work flow
1. create japanese category list -> get category id
    - done!
    - use printCategories to get all the info needed

2. query API with genreid
    a. get number of items
    b. call pagenumber +1

3. insert 100 items to collection
    for each:
    a. check if img link is valid - ig no link the url ends with 'noimage'
    b. check prior existence by item id and by hash
    c. insert item
    d. call segmentation net - args = mongo id, img_url, collection name, category
    e. segmentation worker calls fp with the mask
        i. if err -> deletes from collection
notice that the images are not so so
-> find  strategy to select best img out of 3
    -> first find face -> than crop-> not good! some imgs dont have faces and are good
    -> run all three through the net
        -> for each calculate certainty in each blob
            -> if certainty surpass 80% for more than one -> choose the biggest blob
-> net might not return the right category
    -> in that case we will want to use near categories
        ->e.g, dress = skirt + top
-> max page number is 999
    -> that means that max items returned 99900
-> relevent return fields

->get tracking id from kyle
"""


