"""
playground for testing the recruit API
"""
from .recruit_constants import api_stock, recruitID2generalCategory
from ..constants import db, redis_conn
import logging
from rq import Queue
from time import sleep,time
from .recruit_worker import genreDownloader, GET_ByGenreId, deleteDuplicates
from datetime import datetime
from .dl_excel import mongo2xl
from .fanni import plantForests4AllCategories

today_date = str(datetime.date(datetime.now()))

q = Queue('recruit_worker', connection=redis_conn)
forest = Queue('annoy_forest', connection=redis_conn)


def theArchiveDoorman(col_name):
    """
    clean the archive from items older than a week
    send items to archive
    """
    collection = db[col_name]
    collection_archive = db[col_name+'_archive']
    archiver_indexes = collection_archive.index_information().keys()
    if 'id_1' not in archiver_indexes:
        collection_archive.create_index('id', background=True)
    archivers = collection_archive.find()
    y_new, m_new, d_new = map(int, today_date.split("-"))
    for item in archivers:
        y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
        days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
        if days_out < 7:
            collection_archive.update_one({'id': item['id']}, {"$set": {"status.days_out": days_out}})
        else:
            collection_archive.delete_one({'id': item['id']})

    # add to the archive items which were not downloaded in the last 2 days
    notUpdated = collection.find({"download_data.dl_version": {"$ne": today_date}})
    for item in notUpdated:
        y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
        days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
        if days_out>2:
            collection.delete_one({'id': item['id']})
            existing = collection_archive.find_one({"id": item["id"]})
            if existing:
                continue

            if days_out < 7:
                item['status']['instock'] = False
                item['status']['days_out'] = days_out
                collection_archive.insert_one(item)

    # move to the archive all the items which were downloaded today but are out of stock
    outStockers = collection.find({'status.instock': False})
    for item in outStockers:
        collection.delete_one({'id': item['id']})
        existing = collection_archive.find_one({"id": item["id"]})
        if existing:
            continue
        collection_archive.insert_one(item)

    collection_archive.reindex()

def log2file(LOG_FILENAME):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILENAME, mode= 'w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


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


def API4printing(genreId, gender, category_name, skip, useLog=False, logger=logging):
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

    summery = 'gender: %s, genreId: %s, category_name: %s , total_count: %s, instock: %s, , japanese: %s , %s , %s'\
              % (gender, genreId, category_name, allitems, instock_only, japanese_name, sec_cat, top_cat)
    if useLog:
        logger.info(summery)
    else:
        print(summery)
    return skip


def getCategoryName(genreId):
    tmp = [x for x in api_stock if x['genreId']==genreId]
    if len(tmp)>0:
        category_name = tmp[0]['category_name']
    else:
        category_name = 'new_cat'
    return category_name


def printCategories(only_known=True, useLog=False):
    if useLog:
        logger =log2file('/home/developer/yonti/recruit_categories.log')
    if only_known:
        for cat in api_stock:
            genreId = cat['genreId']
            category_name = cat['category_name']
            gender = cat['gender']
            API4printing(genreId, gender, category_name, 0, useLog=useLog, logger=logger)
        print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxox')
    else:
        for gender in ['Female', 'Male']:
            for m in range(99):
                skip = 0
                for s in range(99):
                    genreId = generate_genreid(gender, m, s)
                    category_name = getCategoryName(genreId)
                    skip = API4printing(genreId, gender, category_name, skip,useLog=useLog, logger=logger)
                    if skip == 3:
                        break
                print('xoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxoxox')

"""
use printCategories to scan the api and print all categories under ladies' and men's fashion
#printCategories(False)
"""


def download_recruit(delete=False):
    s = time()

    for gender in ['Male', 'Female']:
        col_name = 'recruit_'+gender
        collection = db[col_name]
        if delete:
            collection.delete_many({})
        indexes = collection.index_information().keys()
        for idx in ['id', 'img_hash', 'categories', 'images.XLarge', 'download_data.dl_version']:
            idx_1 = idx + '_1'
            if idx_1 not in indexes:
                collection.create_index(idx, background=True)
        status_full_path = 'collections.'+col_name+'.status'
        db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})

    handler = log2file('/home/developer/yonti/recruit_download_stats.log')
    handler.info('download started')
    id_count = len(recruitID2generalCategory)
    for x, genreId in enumerate(recruitID2generalCategory.keys()):
        q.enqueue(genreDownloader, args=([genreId]), timeout=5400)
        print('%d/%d -> %s sent to download worker' %(x,id_count, genreId))
        while q.count > 5:
            sleep(60)

    e = time()
    duration = e-s
    print ('download time : %d' % duration )

    deleteDuplicates()
    for gender in ['Male', 'Female']:
        col_name = 'recruit_' + gender
        status_full_path = 'collections.' + col_name + '.status'
        db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Finishing"}})
        theArchiveDoorman(col_name)
        forest_job = forest.enqueue(plantForests4AllCategories, col_name=col_name, timeout=3600)
        while not forest_job.is_finished and not forest_job.is_failed:
            sleep(300)
        if forest_job.is_failed:
            print ('annoy plant forest failed')

    dl_info = {"date": today_date,
               "dl_duration": duration,
               "store_info": []}

    mongo2xl('recruit_me', dl_info)


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


