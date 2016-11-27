"""
playground for testing the recruit API
"""
import logging
from datetime import datetime
from time import sleep
from time import time

from rq import Queue

from ...constants import db, redis_conn
from ..annoy_dir.fanni import plantForests4AllCategories
from ..general.dl_excel import mongo2xl
from .ebay_API_worker import downloader
from .ebay_constants import sub_attributes

today_date = str(datetime.date(datetime.now()))

q = Queue('ebay_API_worker', connection=redis_conn)
forest = Queue('annoy_forest', connection=redis_conn)


def log2file(log_filename, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def theArchiveDoorman(col):
    """
    clean the archive from items older than a week
    send items to archive
    """
    collection = db[col]
    collection_archive = db[col+'_archive']
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
    notupdated = collection.find({"download_data.dl_version": {"$ne": today_date}})
    for item in notupdated:
        y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
        days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
        if days_out > 2:
            collection.delete_one({'id': item['id']})
            existing = collection_archive.find_one({"id": item["id"]})
            if existing:
                continue

            if days_out < 7:
                item['status']['instock'] = False
                item['status']['days_out'] = days_out
                collection_archive.insert_one(item)

    # move to the archive all the items which were downloaded today but are out of stock
    outstockers = collection.find({'status.instock': False})
    for item in outstockers:
        collection.delete_one({'id': item['id']})
        existing = collection_archive.find_one({"id": item["id"]})
        if existing:
            continue
        collection_archive.insert_one(item)

    collection_archive.reindex()


def download_ebay_api(col, gender, price_bottom=0, price_top=10000, mode=False, reset=False):
    s = time()
    collection = db[col]
    if reset:
        collection.delete_many({})

    indexes = collection.index_information().keys()

    for idx in ['id', 'sku', 'img_hash', 'categories', 'images.XLarge']:
        idx_1 = idx+'_1'
        if idx_1 not in indexes:
            collection.create_index(idx, background=True)

    download_log = '/home/developer/yonti/ebay_'+gender+'_download_stats.log'
    handler = log2file(download_log, 'download')
    handler.info('download started')
    keywords_log = '/home/developer/yonti/keywords_'+gender+'.log'
    handler = log2file(keywords_log, 'keyword')
    handler.info('keyword started')
    for sub_attribute in sub_attributes:
        q.enqueue(downloader, args=(GEO, gender, sub_attribute, price_bottom, price_top, mode), timeout=14400)
        print(sub_attribute + ' sent to download worker')
        sleep(30)
        while q.count > 0:
            sleep(30)
    e = time()
    dl_duration = e-s
    print ('%s download time : %d' % (gender, dl_duration))
    return duration

if __name__ == '__main__':
    # TODO: use argsparse to select GEO
    GEO = 'US'
    duration = 0
    for gen in ['Male', 'Female']:
        col_name = 'ebay_' + GEO + '_' + gen
        status_full_path = 'collections.' + col_name + '.status'
        db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Working"}})
        duration += download_ebay_api(col_name, gen)
        db.download_status.update_one({"date": today_date}, {"$set": {status_full_path: "Finishing"}})

    for gen in ['Male', 'Female']:
        col_name = 'ebay_' + GEO + '_' + gen
        # theArchiveDoorman(col_name)
        forest_job = forest.enqueue(plantForests4AllCategories, col_name=col_name, timeout=3600)
        while not forest_job.is_finished and not forest_job.is_failed:
            sleep(300)
        if forest_job.is_failed:
            print ('annoy plant forest failed')

        print (col_name + "Update Finished!!!")

    dl_info = {"date": today_date,
               "dl_duration": duration,
               "store_info": []}

    mongo2xl('ebay_me', dl_info)
