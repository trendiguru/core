import time

import numpy as np
from rq import Queue
import cv2

from .. import constants

redis_conn = constants.redis_conn

# Tell RQ what Redis connection to use

def paperdoll_enqueue(img_url_or_cv2_array, filename=None, async=True, queue=None, use_tg_worker=True):
    """
    The 'parallel matlab queue' which starts engines and keeps them warm is 'pd'.  This worker should be running somewhere (ideally in a screen like pd1).
    The use_tg_worker argument forces  use/nonuse of the tgworker than knows how to keep the engines warm and can be started along the lines of:
        rqworker -u redis://redis1-redis-1-vm:6379  -w rq.tgworker.TgWorker  pd
    :param img_url_or_cv2_array: the image/url
    :param async: whether to run async or sync
    :param queue: queue name on which to run paperdoll
    :param use_tg_worker: whether or not to use special tg worker, if so queue needs to have been started with -t tgworker
    :return: mask, label_dict, pose
    """
    if queue is None:
        # this is the one that has persistent matlab engines, requires get_parse_mask_parallel and workers on that queue that have been started
        # using: rqworker pd -w rq.tgworker.TgWorker
        if use_tg_worker:
            queue_name = constants.parallel_matlab_queuename
            queue = Queue(queue_name, connection=redis_conn)
            job1 = queue.enqueue('trendi_guru_modules.paperdoll.pd.get_parse_mask_parallel', img_url_or_cv2_array,
                                 filename=filename)
        else:
            queue_name = constants.nonparallel_matlab_queuename
            queue = Queue(queue_name, connection=redis_conn)
            job1 = queue.enqueue('trendi_guru_modules.paperdoll.pd.get_parse_mask',img_url_or_cv2_array)
    print('started pd job on queue:'+str(queue))
    start = time.time()
    if not async:
        print('running synchronously (waiting for result)'),
        while job1.result is None:
            time.sleep(0.5)
            print('.'),
            elapsed_time = time.time()-start
            if elapsed_time>constants.paperdoll_ttl :
                print('timeout waiting for pd.get_parse_mask')
                return
        print('')
    else:
        print('running asynchronously (not waiting for result)')
    return job1



def show_parse(filename=None, img_array=None):
    if filename is not None:
        img_array = cv2.imread(filename)
    if img_array is not None:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
        maxVal = 31  # 31 categories in paperdoll
        scaled = np.multiply(img_array, int(255 / maxVal))
        dest = cv2.applyColorMap(scaled, cv2.COLORMAP_RAINBOW)
        print('writing parse_img.jpg',img_array)
        cv2.imwrite('parse_img.jpg',img_array)
        cv2.imshow("dest", dest)
        cv2.waitKey(0)


def show_max(parsed_img, labels):
    maxpixval = np.ma.max
    print('max pix val:' + str(maxpixval))
    maxlabelval = len(labels)
    print('max label val:' + str(maxlabelval))




import time
def callback_example(queue_name,previous_job_id,*args,**kwargs):
    print('this is the callback calling')
    if previous_job_id is None:
        logging.debug('got no previous job id')
        return
    connection = constants.redis_conn
    if connection is None:
        logging.debug('got no redis conn')
        return
    queue = Queue(queue_name, connection=connection)
    if queue is None:
        logging.debug('got no queue')
        return
    job1_answers = queue.fetch_job(previous_job_id)
    print('prev result:')
    print job1_answers

    logging.warning('this is the callback calling')
    return (567,job1_answers)

    #run a parallelization test
if __name__ == "__main__":

    img,labels,pose=paperdoll_enqueue('http://clothingparsing.com/sessions/ff57f475a232acfc1979d9aa2ad161afe6b9c91b/image.jpg',async=False)
    show_parse(img_array=img)

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
