import re
from datetime import datetime
from time import sleep, time

from ..general.db_utils import log2file, print_error, thearchivedoorman, progress_bar, refresh_similar_results
from rq import Queue
import pymongo
from ...constants import db, redis_conn
from ..annoy_dir.fanni import plantAnnoyForest, reindex_forest
from ..general.dl_excel import mongo2xl
from .amazon_constants import plus_sizes, amazon_categories_list

last_pct = 0
today_date = str(datetime.date(datetime.now()))
forest = Queue('annoy_forest', connection=redis_conn)


def clear_duplicates(col_name):
    global last_pct
    collection = db[col_name]
    all_items = collection.find({}, {'id': 1, 'parent_asin': 1, 'img_hash': 1, 'images.XLarge': 1, 'sizes': 1,
                                     'color': 1, 'p_hash': 1},
                                no_cursor_timeout=True, cursor_type=pymongo.CursorType.EXHAUST).sort({'$natural':1})
    bef = all_items.count()
    block_size = bef/100
    for i, item in enumerate(all_items):
        # m, r = divmod(i, block_size)
        # if r == 0:
        #     last_pct = progress_bar(block_size, bef, m, last_pct)
        print i
        item_id = item['_id']
        keys = item.keys()
        if any(x for x in ['id', 'parent_asin', 'img_hash', 'images', 'sizes', 'color', 'p_hash'] if x not in keys):
            # collection.delete_one({'_id':item_id})
            continue
        idx = item['id']
        collection.delete_many({'id': idx, '_id': {'$ne': item_id}})

        parent = item['parent_asin']
        parent_exists = collection.find({'parent_asin': parent, '_id': {'$ne': item_id}},
                                        {'id': 1, 'sizes': 1, 'color': 1})
        if parent_exists:
            id_to_del = []
            current_sizes = item['sizes']
            current_color = item['color']
            for tmp_item in parent_exists:
                tmp_id = tmp_item['_id']
                tmp_color = tmp_item['color']
                if tmp_color != current_color:
                    continue
                tmp_sizes = tmp_item['sizes']
                for size in tmp_sizes:
                    if size not in current_sizes:
                        current_sizes.append(size)
                id_to_del.append(tmp_id)
            if len(id_to_del):
                collection.delete_many({'_id': {'$in': id_to_del}})

        img_hash = item['img_hash']
        collection.delete_many({'img_hash': img_hash, '_id': {'$ne': item_id}})

        p_hash = item['p_hash']
        collection.delete_many({'p_hash': p_hash, '_id': {'$ne': item_id}})

        img_url = item['images']['XLarge']
        collection.delete_many({'images.XLarge': img_url, '_id': {'$ne': item_id}})
    all_items.close()
    print('')
    print_error('CLEAR DUPLICATES', 'count before : %d\ncount after : %d' % (bef, collection.count()))


def update_drive(col, cc, items_before=None, dl_duration=None):
    items_after = items_new = items_scanned = 0
    for gender in ['Female', 'Male']:
        if col == 'amazon':
            col_name = '%s_%s_%s' %(col, cc, gender)
        else:
            col_name = '%s_%s' % (col, gender)
        collection = db[col_name]
        items_after += collection.count()
        items_new += collection.count({'download_data.first_dl': today_date})
        items_scanned += collection.count({'download_data.dl_version': today_date})
    dl_duration = dl_duration or 'daily-update'
    items_before = items_before or 'daily-update'
    dl_info = {"start_date": today_date,
               "dl_duration": dl_duration,
               "items_before": items_before,
               "items_after": items_after,
               "items_new": items_new,
               'items_scanned': items_scanned}
    mongo2xl(col_name, dl_info)


def daily_annoy(col_name, categories, all_cats=False):
    collection = db[col_name]
    if not all_cats:
        categories_with_changes = []
        for cat in categories:
            if collection.count({'categories': cat, 'download_data.first_dl': today_date}) > 0:
                categories_with_changes.append(cat)
                print('%s will be re-annoyed' % cat)
        categories = categories_with_changes

    jobs = []
    for c, cat in enumerate(categories):
        forest_job = forest.enqueue(plantAnnoyForest, args=(col_name, cat, 250), timeout=3600)
        jobs.append({'cat': cat, 'job': forest_job, 'running': True})

    while any(job['running'] for job in jobs if job['running']):
        for job in jobs:
            if job['job'].is_failed:
                print ('annoy for %s failed' % job['cat'])
                job['running'] = False
                jobs = [x for x in jobs if x['running']]
            elif job['job'].is_finished:
                msg = "%s annoy done!" % (job['cat'])
                print_error(msg)
                job['running'] = False
                jobs = [x for x in jobs if x['running']]
            else:
                print '#',
                sleep(15)

    reindex_forest(col_name)
    return categories


def verify_plus_size(size_list):
    splited_list = []
    for size in size_list:
        size_upper = size.upper()
        split = re.split(r'\(|\)| |-|,', size_upper)
        for s in split:
            splited_list.append(s)
    if 'SMALL' in splited_list:
        return False
    return any(size for size in splited_list if size in plus_sizes)


def update_plus_size_collection(gender, categories, cc='US', skip_refresh=False):
    amaze_start = time()
    amaze_name = 'amaze_%s' % gender
    amaze = db[amaze_name]
    items_before = 0
    for gen in ['Female', 'Male']:
        col_name = '%s_%s' % ('amaze', gen)
        items_before += db[col_name].count()
    amazon_name = 'amazon_%s_%s' % (cc, gender)
    amazon = db[amazon_name].find(no_cursor_timeout=True)
    amazon_total = amazon.count()
    inserted = 0
    for x, item in enumerate(amazon):
        if x % 100 == 0:
            print('%d/%d' % (x, amazon_total))
        idx = item['id']
        # check if already exists in plus collection
        exists = amaze.count({'id': idx})
        if exists:
            continue
        sizes = item['sizes']
        if type(sizes) != list:
            sizes = [sizes]

        its_plus_size = verify_plus_size(sizes)
        if its_plus_size:
            inserted += 1
            if inserted % 100 == 0:
                print ('so far %s inserted' % inserted)
            amaze.insert_one(item)
    amazon.close()
    clear_duplicates(amaze_name)  # add status bar
    thearchivedoorman(amaze_name, instock_limit=30, archive_limit=60)
    print_error('ARCHIVE DOORMAN FINISHED')

    updated_categories = daily_annoy(amaze_name, categories, True)
    if not skip_refresh:
        refresh_similar_results('amaze', updated_categories)

    amaze_end = time()
    dl_duration = amaze_end - amaze_start
    update_drive('amaze', cc, items_before, dl_duration)


def daily_amazon_updates(col_name, gender, all_cats=False, cc='US', skip_refresh=False):
    # redo annoy for categories which has been changed
    daily_annoy(col_name, amazon_categories_list, all_cats)

    # refresh items which has been changed
    if not skip_refresh:
        refresh_name = 'amazon_%s' % cc
        refresh_similar_results(refresh_name, amazon_categories_list)

    # upload file to drive
    update_drive('amazon', cc)

    col_upper = col_name.upper()
    print_error('%s DOWNLOAD FINISHED' % col_upper)

    # update plus size
    if cc == 'US':
        update_plus_size_collection(gender, amazon_categories_list, cc, skip_refresh)
        plus = col_upper + ' PLUS SIZE'
        print_error('%s FINISHED' % plus)
    return


def post_download(col_name, gender, cc, log_name):
    clear_duplicates(col_name)  # add status bar
    thearchivedoorman(col_name, instock_limit=30, archive_limit=60)
    print_error('ARCHIVE DOORMAN FINISHED')

    message = '%s is Done!' % col_name
    log2file(mode='a', log_filename=log_name, message=message, print_flag=True)
    # after download finished its time to build a new annoy forest

    daily_amazon_updates(col_name, gender, all_cats=True, cc=cc)

    notes_full_path = 'collections.' + col_name + '.notes'
    status_full_path = 'collections.' + col_name + '.status'

    db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Done",
                                                                  notes_full_path: 'so and so'}})