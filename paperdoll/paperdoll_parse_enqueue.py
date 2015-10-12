import time
import numpy as np
from rq import Queue
from redis import Redis
import cv2

from trendi_guru_modules import constants

redis_conn = Redis()

# Tell RQ what Redis connection to use

def paperdoll_enqueue(img_url_or_cv2_array, async=True,queue=None,use_tg_worker=True,callback_function='testfunc',*callback_function_args,**callback_function_kwargs):
    if queue is None:
        if use_tg_worker:   #this is the one that has persistent matlab engines, requires get_parse_mask_parallel and workers on that queue that have been started
                            # using: rqworker pd -w rq.tgworker.TgWorker
            queue = Queue('pd', connection=redis_conn)
            job = queue.enqueue('pd.get_parse_mask_parallel', img_url_or_cv2_array,callback_function,callback_function_args,callback_function_kwargs)
        else:
            queue = Queue('pd_nonparallel', connection=redis_conn)
            job = queue.enqueue('pd.get_parse_mask',img_url_or_cv2_array,callback_function,callback_function_args,callback_function_kwargs))
    print('started pd job on queue:'+str(queue))
    start = time.time()
    if not async:
        print('running synchronously'),
        while job.result is None:
            time.sleep(0.5)
            print('.'),
            elapsed_time = time.time()-start
            if elapsed_time>constants.paperdoll_ttl:
                print('timeout waiting for pd.get_parse_mask')
                return
        print('')
        return job.result
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


#run a parallelization test
if __name__ == "__main__":
    urls = ['http://i.imgur.com/ahFOgkm.jpg',\
           'http://www.wantdresses.com/wp-content/uploads/2015/09/group-of-vsledky-obrzk-google-pro-httpwwwoblectesecz-awesome-prom-dresses.jpg',\
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
       # n = paperdoll_enqueue(url, async = True,queue=queue)
        img, labels, pose = paperdoll_enqueue(url, async = True,use_tg_worker=True)
#        print('labels:'+str(labels))
#        print('')
    elapsed_time = time.time() - start_time
    print('tot elapsed:'+str(elapsed_time)+',per image:'+str(float(elapsed_time)/len(urls)))

        #        show_max(img, labels)
#        show_parse(img_array=img)


