import logging

import numpy as np
from rq import Queue
import cv2

from trendi import constants


redis_conn = constants.redis_conn
TTL = constants.general_ttl
# Tell RQ what Redis connection to use

def paperdoll_enqueue(img_url_or_cv2_array, person_id=None, async=True, queue_name=None, use_tg_worker=True,
                      use_parfor=False, at_front=False):
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
    if queue_name is None:
        if use_tg_worker:
            queue_name = constants.parallel_matlab_queuename
        else:
            queue_name = constants.nonparallel_matlab_queuename
    if use_parfor:
        queue_name = 'pd_parfor'
    queue = Queue(queue_name, connection=redis_conn)
    job1 = queue.enqueue_call(func='trendi.paperdoll.pd.get_parse_mask_parallel',
                              args=(img_url_or_cv2_array, person_id),
                              ttl=TTL, result_ttl=TTL, timeout=TTL, at_front=at_front)
    if isinstance(img_url_or_cv2_array, basestring):
        url = img_url_or_cv2_array
    else:
        url = None
    print('started pd job on queue:'+str(queue)+' url:'+str(url))
    start = time.time()
    if not async:
        print('running synchronously (waiting for result)'),
        while job1.result is None:
            time.sleep(0.5)
            print('.'),
            elapsed_time = time.time()-start
            if elapsed_time>constants.paperdoll_ttl :
                if isinstance(img_url_or_cv2_array,basestring):
                    print('timeout waiting for pd.get_parse_mask, url='+img_url_or_cv2_array)
                    raise Exception('paperdoll timed out on this file:',img_url_or_cv2_array)
                else:
                    print('timeout waiting for pd.get_parse_mask, img_arr given')
                    raise Exception('paperdoll timed out with an img arr')
            return
        print('')
        print('elapsed time in paperdoll_enqueue:'+str(elapsed_time))
        #todo - see if I can return the parse, pose etc here without breaking anything (aysnc version gets job1 back so there may be expectation of getting a job instead of tuple
    else:
        print('running asynchronously (not waiting for result)')
    return job1

def show_parse(filename=None, img_array = None,save=False):
    if filename is not None:
        img_array = cv2.imread(filename)
    if img_array is not None:
#        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_array)
        img_array = img_array-1
        maxVal = 55  # 31 categories in paperdoll
        scaled = np.multiply(img_array, int(255 / maxVal))
        dest = cv2.applyColorMap(scaled, cv2.COLORMAP_HOT)
#        print('writing parse_img.jpg',img_array)
#        cv2.imwrite('parse_img.jpg',img_array)

        h,w = img_array.shape[:2]
        print('h {0} w {1}'.format(h,w))
#        new_image = np.zeros([h,2*w])
        cv2.imshow("orig", img_array)
        cv2.imshow("dest", dest)
        newfilename = filename.split("pdniceout.bmp")[0]+'.jpg'
        cv2.imwrite(newfilename,dest)
        cv2.waitKey(0)

def colorbars(max=55):
    bar_height = 10
    bar_width = 20
    new_img = np.ones([max*bar_height,bar_width],np.uint8)
    for i in range(0,max):
        new_img[i*bar_height:(i+1)*bar_height,:] = int(i*255.0/max)
    #print(new_img)
    cv2.imwrite('testvarout.jpg',new_img)
    print('writing file')
 #   cv2.imshow('colorbars',new_img)
 #   cv2.waitKey(0)

    show_parse(img_array=new_img+1)

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

    print 'this is the callback calling'
    return 567, job1_answers

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

    for url in urls:
        i+=1
        print('url #'+str(i)+' '+url)
        img, labels, pose = paperdoll_enqueue(url, async = True,use_tg_worker=True)
#        print('labels:'+str(labels))
#        print('')
    elapsed_time = time.time() - start_time
#for timing test see unit test
#    print('tot elapsed:'+str(elapsed_time)+',per image:'+str(float(elapsed_time)/len(urls)))

        #        show_max(img, labels)
#        show_parse(img_array=img)

#box colors from Yang2011 pose estimato code(this is what pd uses)
#yellow magenta cyan red green blue white black

#{'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
#from images here http://www.ics.uci.edu/~dramanan/software/pose/ it seems to correspond to :
#head, head
#chest
#left shoulder, upper arm, lower arm, hand (mayb backwards) (left for viewer, actually right)
#chest, chest, chest
#left hip, knee, ankle, foot (maybe backwards)
#chest
#right shoulder, upper arm, lower arm, hand (maybe backwards)
#chest chest chest
#right hip, knee, ankle, foot (maybe backwards)
#26 boxes * 4 coords/box=104 coords

#right hip, knee, ankle, foot
#
