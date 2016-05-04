__author__ = 'jeremy'

import logging

import numpy as np
from rq import Queue
import cv2
import time
from trendi import constants
from trendi import Utils
from trendi.utils import imutils

redis_conn = constants.redis_conn
TTL = constants.general_ttl
# Tell RQ what Redis connection to use

def nn_doorman_enqueue(img_url_or_cv2_array,async=False):
    """
    The 'doorman queue' which starts engines and keeps them warm is 'neurodoor'.  This worker should be running somewhere (ideally in a screen named something like 'doorman').
    :param img_url_or_cv2_array: the image/url
    """
    queue_name = constants.neurodoorman_queuename
    queue = Queue(queue_name, connection=redis_conn)
    job1 = queue.enqueue_call(func='trendi.nn_doorman.none_may_pass',
                              args=(img_url_or_cv2_array),
                              ttl=TTL, result_ttl=TTL, timeout=TTL)
    if isinstance(img_url_or_cv2_array, basestring):
        url = img_url_or_cv2_array
    else:
        url = None
    print('started doorman job on queue:'+str(queue)+' url:'+str(url))
    start = time.time()
    if not async:
        print('running synchronously (waiting for result)'),
        elapsed_time=0.0
        while job1.result is None:
            time.sleep(0.1)
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


def none_may_pass(caffenet, img_url_or_cv2_array):
    start_time=time.time()
    img = Utils.get_cv2_img_array(img_url_or_cv2_array)
    img_ok = Utils.image_big_enough(img)
    dims = [227,227]

    if img_ok:
        img = imutils.resize_keep_aspect(img, output_file=None, output_size = dims,use_visual_output=False):
        print('image size:'+str(img.shape))
        if len(img.shape) != 3:
            print('got 1-chan image, skipping')
            continue
        elif img.shape[2] != 3:
            print('got n-chan image, skipping - shape:'+str(in_.shape))
            continue
        img = img[:,:,::-1]
        img -= np.array((104.0,116.7,122.7))
        img = img.transpose((2,0,1))
        # shape for input (data blob is N x C x H x W), set data
  #      caffenet.blobs['data'].reshape(1, *in_.shape)
  #      caffenet.blobs['data'].data[...] = in_
        # run net and take argmax for prediction


        out = caffenet.forward_all(data=np.asarray([img]))
        results = int(out['prob'][0])
        plabel = int(out['prob'][0].argmax(axis=0))
        print('results {} label {}'.format(results,plabel))

#        net.forward()
#        out = net.blobs['score'].data[0].argmax(axis=0)
#        result = Image.fromarray(out.astype(np.uint8))
    #        outname = im.strip('.png')[0]+'out.bmp'
#            outname = os.path.basename(imagename)
 #       outname = outname.split('.jpg')[0]+'.bmp'
  #      outname = os.path.join(out_dir,outname)
   #     print('outname:'+outname)
    #        result.save(outname)
    #        fullout = net.blobs['score'].data[0]
        elapsed_time=time.time()-start_time
        print('elapsed time:'+str(elapsed_time)+' tpi:'+str(elapsed_time/len(images)))
        return {''}
    else:
        if img is None:
            raise ValueError("input image is empty")
        elif not img_ok:
            raise ValueError("input image is  too small")
        else:
            raise ValueError("problem writing "+str(filename)+" in get_parse_mask_parallel")
