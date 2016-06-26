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


def log2file(LOG_FILENAME):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_FILENAME, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def download_ebay_API(GEO, gender):
    s = time()
    col = 'ebay_'+gender+'_'+GEO
    db[col].delete_many({})
    log_filename = '/home/developer/yonti/ebay_'+gender+'_download_stats.log'
    handler = log2file(log_filename)
    handler.info('download started')
    for sub_attribute in sub_attributes:
        q.enqueue(downloader, args=(GEO, gender, sub_attribute), timeout=5400)
        print(sub_attribute + ' sent to download worker')
        while q.count > 0:
            sleep(30)
    e = time()
    print ('%s download time : %d' % (gender, e-s))


if __name__=='__main__':
    #TODO: use argsparse to select GEO
    download_ebay_API("US", 'Female')
    download_ebay_API("US", 'Male')



