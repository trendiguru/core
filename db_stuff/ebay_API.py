"""
playground for testing the recruit API
"""
from .ebay_constants import sub_attributes
from ..constants import db, redis_conn
import logging
from rq import Queue
from time import sleep
from .ebay_API_worker import downloader
from time import time
q = Queue('ebay_API_worker', connection=redis_conn)


def log2file(log_filename, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def download_ebay_API(GEO, gender):
    s = time()
    col = 'ebay_'+gender+'_'+GEO
    db[col].delete_many({})
    download_log = '/home/developer/yonti/ebay_'+gender+'_download_stats.log'
    handler = log2file(download_log, 'download')
    handler.info('download started')
    keywords_log = '/home/developer/yonti/keywords.log'
    handler = log2file(keywords_log, 'keyword')
    handler.info('keyword started')
    for sub_attribute in sub_attributes:
        q.enqueue(downloader, args=(GEO, gender, sub_attribute), timeout=5400)
        print(sub_attribute + ' sent to download worker')
        sleep(30)
        while q.count > 0:
            sleep(30)
    e = time()
    print ('%s download time : %d' % (gender, e-s))


if __name__=='__main__':
    #TODO: use argsparse to select GEO
    download_ebay_API("US", 'Female')
    download_ebay_API("US", 'Male')



