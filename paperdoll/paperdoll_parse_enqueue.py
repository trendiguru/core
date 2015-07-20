import requests

from rq import Queue
from redis import Redis
import cv2

import pd


def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())


# Tell RQ what Redis connection to use
def paperdoll_enqueue(img_url):
    redis_conn = Redis()
    q = Queue('jeremyTest', connection=redis_conn)
    # q = Queue('jeremyTest', connection=redis_conn, async=False)  # not async
    # q = Queue(connection=redis_conn)  # no args implies the default queue

# Delay execution of count_words_at_url('http://nvie.com')
    job = q.enqueue(pd.get_parse_mask, img_url)
    # job = q.enqueue(count_words_at_url, 'http://nvie.com')


# print job.result  # => None
# Now, wait a while, until the worker is finished
 #   time.sleep(2)
# print job.result  # => 889

def show_parse(filename=None, img_array=None):
    if filename is not None:
        img_array = cv2.imread(filename)
    if img_array is not None:
        cv2.imshow('img', img_array)
        cv2.waitKey(0)

        # stripped_name=image_url.split('//')[1]
        #    modified_name=stripped_name.replace('/','_')

        # enqueue()