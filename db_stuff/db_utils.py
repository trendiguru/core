import logging
import hashlib
from ..constants import db
from datetime import datetime
from ..Yonti import pymongo_utils


today_date = str(datetime.date(datetime.now()))


def print_error(title, message=''):
    dotted_line = '------------------------------------%s-----------------------------------------' % title

    if len(message) > 0:
        print ('\n%s' % dotted_line)
        print (message)
        print ('%s\n' % dotted_line)
    else:
        print ('\n%s\n' % dotted_line)


def log2file(mode, log_filename, message='', print_flag=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename, mode=mode)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if len(message):
        logger.info(message)
        logger.removeHandler(handler)
        del logger, handler
        if print_flag:
            print_error(message)
    else:
        return logger, handler


def get_hash(image):
    m = hashlib.md5()
    m.update(image)
    url_hash = m.hexdigest()
    return url_hash


def theArchiveDoorman(col_name):
    """
    clean the archive from items older than a week
    send items to archive
    """
    collection = db[col_name]
    archive_name = col_name+'_archive'
    pymongo_utils.delete_or_and_index(collection_name=archive_name, index_list=['id'])
    collection_archive = db[archive_name]
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