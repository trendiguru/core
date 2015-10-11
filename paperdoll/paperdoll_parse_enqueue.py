import time
import numpy as np
from rq import Queue
from redis import Redis
import cv2

from trendi_guru_modules import constants

redis_conn = Redis()

# Tell RQ what Redis connection to use


def paperdoll_enqueue(img_url_or_cv2_array, async=True,queue=None,use_tg_worker=True):
    if(use_tg_worker):
        return paperdoll_enqueue_parallel(img_url_or_cv2_array=img_url_or_cv2_array,async=async)
    else:
        if queue is None:
            queue = Queue('paperdoll', connection=redis_conn)
        print('starting pd job on queue:'+str(queue))
        job = queue.enqueue('pd.get_parse_mask', img_url_or_cv2_array=img_url_or_cv2_array)
        start = time.time()
        if not async:
            while job.result is None:
                time.sleep(0.5)
                elapsed_time = time.time()-start
                if elapsed_time>constants.paperdoll_ttl:
                    print('timeout waiting for pd.get_parse_mask')
                    return [[],[],[]]
            return job.result
        return [job.result,None,None]


def paperdoll_enqueue_parallel(img_url_or_cv2_array,async=True):
    qp = Queue('pd_parallel', connection=redis_conn)
    print('starting pd job on parallel queue:'+str(qp))
    job = qp.enqueue('pd.get_parse_mask_parallel', img_url_or_cv2_array)
    start = time.time()
    if not async:
        print('running async'),
        while job.result is None:
            time.sleep(0.5)
            print('.'),
            elapsed_time = time.time()-start
            if elapsed_time>constants.paperdoll_ttl:
                print('timeout waiting for pd.get_parse_mask')
                return [[],[],[]]
        return job.result
    #the caller expects three results...
    return [job.result,None,None]

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


#run a timing test
if __name__ == "__main__":
    urls = ['http://i.imgur.com/ahFOgkm.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/sexy-plus-sized-prom-dresses-at-peaches-boutique-peaches-awesome-prom-dresses.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/07/skvdfw-l.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/xoxo-prom-dresses-awesome-prom-dresses.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/prom-dresses-on-pinterest-orange-prom-dresses-pink-prom-dresses-awesome-prom-dresses.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/fun-prom-dresses-2013-look-awesome-in-ombre-blog-at-the-prom-awesome-prom-dresses.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/dress-jewellery-picture-more-detailed-picture-about-hot-sale-awesome-prom-dresses.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/07/rs_634x926-140402114112-634-8Prom-Dress-ls.4214.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/group-of-vsledky-obrzk-google-pro-httpwwwoblectesecz-awesome-prom-dresses.jpg',\
            'http://www.wantdresses.com/wp-content/uploads/2015/09/gowns-blue-picture-more-detailed-picture-about-awesome-strapless-awesome-prom-dresses.jpg']
    i = 0
    start_time = time.time()
    queue = Queue('paperdoll_test', connection=redis_conn)

    for url in urls:
        i+=1
        print('url #'+str(i)+' '+url)
    #    img, labels, pose = paperdoll_enqueue(url, async = True,queue=queue)
        n = paperdoll_enqueue(url, async = True,queue=queue)
       # img, labels, pose = paperdoll_enqueue_parallel(url, async = True)
#        print('labels:'+str(labels))
#        print('')
    elapsed_time = time.time() - start_time
    print('tot elapsed:'+str(elapsed_time)+',per image:'+str(float(elapsed_time)/len(urls)))

        #        show_max(img, labels)
#        show_parse(img_array=img)


