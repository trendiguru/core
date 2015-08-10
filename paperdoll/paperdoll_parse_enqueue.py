import requests
import time
import numpy as np

from rq import Queue
from redis import Redis
import cv2


def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())


# Tell RQ what Redis connection to use

def paperdoll_enqueue(img_url, async=True):
    redis_conn = Redis()
    q = Queue('jeremyTest', connection=redis_conn)
    # q = Queue('jeremyTest', connection=redis_conn, async=False)  # not async
    # q = Queue(connection=redis_conn)  # no args implies the default queue

# Delay execution of count_words_at_url('http://nvie.com')
    job = q.enqueue('pd.get_parse_mask', image_url=img_url)
    # job = q.enqueue(count_words_at_url, 'http://nvie.com')
    if not async:
        while job.result is None:
            time.sleep(0.5)
    return job.result


# print job.result  # => None
# Now, wait a while, until the worker is finished
 #   time.sleep(2)
# print job.result  # => 889

def show_parse(filename=None, img_array=None):
    if filename is not None:
        img_array = cv2.imread(filename)
    if img_array is not None:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
        maxVal = 31  # 31 categories in paperdoll
        scaled = np.multiply(img_array, int(255 / maxVal))
        dest = cv2.applyColorMap(scaled, cv2.COLORMAP_RAINBOW)
        cv2.imshow("dest", dest)
        cv2.waitKey(0)


        # cv2.imshow('img', img_array)
        #        cv2.waitKey(0)

        # stripped_name=image_url.split('//')[1]
        #    modified_name=stripped_name.replace('/','_')

        # enqueue()


def show_max(parsed_img, labels):
    maxpixval = np.ma.max
    print('max pix val:' + str(maxpixval))
    maxlabelval = len(labels)
    print('max label val:' + str(maxlabelval))


if __name__ == "__main__":
    img, labels, pose = paperdoll_enqueue('image.jpg', async=False)
    show_max(img, labels)
    show_parse(img_array=img)