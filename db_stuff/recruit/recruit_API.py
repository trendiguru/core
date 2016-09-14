from datetime import datetime
from time import sleep,time

import argparse

from ..general.db_utils import log2file, thearchivedoorman, refresh_similar_results
from rq import Queue

from ...constants import db, redis_conn
from ..annoy_dir.fanni import plantForests4AllCategories
from ..general.dl_excel import mongo2xl
from .recruit_constants import api_stock, recruitID2generalCategory
from .recruit_worker import genreDownloader, GET_ByGenreId, deleteDuplicates

today_date = str(datetime.date(datetime.now()))

q = Queue('recruit_worker', connection=redis_conn)
forest = Queue('annoy_forest', connection=redis_conn)


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

    summery = 'gender: %s, genreId: %s, category_name: %s , total_count: %s, instock: %s, , japanese: %s , %s , %s'\
              % (gender, genreId, category_name, allitems, instock_only, japanese_name, sec_cat, top_cat)

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
        log2file(mode='w', log_filename='/home/developer/yonti/recruit_categories.log')
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


def post_download(duration=0, before_count=0, full=True):
    deleteDuplicates()
    after_count = 0
    new_count = 0
    for gender in ['Male', 'Female']:
        col_name = 'recruit_' + gender
        collection = db[col_name]
        after_count += collection.count()
        new_count += collection.find({'download_data.first_dl': today_date}).count()
        if full:
            status_full_path = 'collections.' + col_name + '.status'
            db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Finishing"}})
        thearchivedoorman(col_name)
        forest_job = forest.enqueue(plantForests4AllCategories, col_name=col_name, timeout=3600)
        while not forest_job.is_finished and not forest_job.is_failed:
            sleep(300)
        if forest_job.is_failed:
            print ('annoy plant forest failed')

    dl_info = {"start_date": today_date,
               "dl_duration": duration,
               "items_before": before_count,
               "items_after": after_count,
               "items_new": new_count}

    mongo2xl('recruit_me', dl_info)

    if full:
        for gender in ['Male', 'Female']:
            col_name = 'recruit_' + gender
            collection = db[col_name]
            new_items = collection.find({'download_data.first_dl': today_date}).count()
            status_full_path = 'collections.' + col_name + '.status'
            notes_full_path = 'collections.' + col_name + '.notes'
            db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Done",
                                                                 notes_full_path: new_items}})

    refresh_similar_results('recruit')


def download_recruit(delete=False):
    s = time()
    before_count = 0
    for gender in ['Male', 'Female']:
        col_name = 'recruit_'+gender
        collection = db[col_name]
        before_count += collection.count()
        # if delete:
        #     collection.delete_many({})
        # indexes = collection.index_information().keys()
        # for idx in ['id', 'img_hash', 'categories', 'images.XLarge', 'download_data.dl_version']:
        #     idx_1 = idx + '_1'
        #     if idx_1 not in indexes:
        #         collection.create_index(idx, background=True)
        status_full_path = 'collections.'+col_name+'.status'
        db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})

    log2file(mode='w', log_filename='/home/developer/yonti/recruit_download_stats.log', message='download started' )
    id_count = len(recruitID2generalCategory)
    for x, genreId in enumerate(recruitID2generalCategory.keys()):
        q.enqueue(genreDownloader, args=([genreId]), timeout=10000)
        print('%d/%d -> %s sent to download worker' %(x,id_count, genreId))
        while q.count > 5:
            sleep(60)

    e = time()
    duration = e-s
    print ('download time : %d' % duration )
    return duration, before_count


def get_user_input():
    parser = argparse.ArgumentParser(description='"@@@ Recruit Download @@@')
    parser.add_argument('-p', '--post', dest="only_post", default=False, action='store_true',
                        help='only run post download steps')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    user_inputs = get_user_input()
    only_post = user_inputs.only_post
    if only_post:
        post_download(full=False)
    else:
        d, b = download_recruit()
        post_download(d, b)


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


