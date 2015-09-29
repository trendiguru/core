import requests
import time
import numpy as np

from rq import Queue
from redis import Redis
import cv2


redis_conn = Redis()
q = Queue('pd', connection=redis_conn)

# Tell RQ what Redis connection to use

def paperdoll_enqueue(img_url_or_cv2_array, async=True):
    job = q.enqueue('pd.get_parse_mask', img_url_or_cv2_array=img_url_or_cv2_array)
    if not async:
        while job.result is None:
            time.sleep(0.5)
    return job.result

def paperdoll_enqueue_parallel(img_url_or_cv2_array,async=True):
    qp = Queue('pd_parallel', connection=redis_conn)
    job = qp.enqueue('pd.get_parse_mask_parallel', img_url_or_cv2_array=img_url_or_cv2_array)
    if not async:
        while job.result is None:
            time.sleep(0.5)
    return job.result

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


def show_max(parsed_img, labels):
    maxpixval = np.ma.max
    print('max pix val:' + str(maxpixval))
    maxlabelval = len(labels)
    print('max label val:' + str(maxlabelval))


if __name__ == "__main__":
    url = 'http://i.imgur.com/ahFOgkm.jpg'
    img, labels, pose = paperdoll_enqueue(url, async=False)
    show_max(img, labels)
    show_parse(img_array=img)


