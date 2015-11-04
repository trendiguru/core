__author__ = 'jeremy'

import time
from rq import Queue
import cv2
import logging
from . import constants
import Utils
import os
redis_conn = constants.redis_conn

# Tell RQ what Redis connection to use

def caffe_general(url_or_img_array, async=True,filename=None):
    """
        this is supposed to take an image and give a caffe nn result. make sure someone has run
        rqworker -u redis://redis1-redis-1-vm:6379 caffe_general
    :param img_url_or_cv2_array: the image/url
    :param async: whether to run async or sync
    :return: dict like {class1:probability1, class2:probability2}
    """
    delete_when_done = False
    img_arr = Utils.get_cv2_img_array(url_or_img_array)
    filename = filename or (rand_string()+'.jpg')
    if img_arr is not None and cv2.imwrite(filename, img_arr):
        queue_name = 'caffe_general'
        queue = Queue(queue_name, connection=redis_conn)
        job = queue.enqueue(ask_caffe,filename)
        print('started caffe job on queue:'+str(queue))
        start = time.time()
        if not async:
            print('running synchronously (waiting for result)'),
            while job.result is None:
                time.sleep(0.5)
                print('.'),
                time_elapsed = time.time()-start
                if time_elapsed>constants.caffe_general_ttl :
                    print('timeout waiting for caffe_general')
                    return None
            print('')
            if delete_when_done:
                try:
                    os.remove(filename)
                except Exception as e:
                    logging.warning("caffe_general could not delete file {0} because of exception: \n{1}".format(filename, e))
            print(job.result)
            return job.result
        else:
            print('running asynchronously (not waiting for result)')
        if delete_when_done:
            try:
                os.remove(filename)
            except Exception as e:
                logging.warning("caffe_general could not delete file {0} because of exception: \n{1}".format(filename, e))
        return job

            return
    else:
        raise ValueError("either image is empty or problem writing")


def ask_caffe(filename):
    caffe_path = constants.caffe_path_in_container
    bin_path = os.path.join(caffe_path,'build/examples/cpp_classification/classification.bin')
    prototxt = os.path.join(caffe_path,'models/bvlc_reference_caffenet/deploy.prototxt')
    caffemodel = os.path.join(caffe_path,'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    binproto = os.path.join(caffe_path,'/data/ilsvrc12/imagenet_mean.binaryproto')
    words = os.path.join(caffe_path,'/data/ilsvrc12/synset_words.txt')
#./build/examples/cpp_classification/classification.bin ./models/bvlc_reference_caffenet/deploy.prototxt ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel ./data/ilsvrc12/imagenet_mean.binaryproto ./data/ilsvrc12/synset_words.txt ./examples/images/cat.jpg
    detect_command = "{bin_path} {prototxt} {caffemodel} {binproto} {words} {img}" \
        .format(bin_path=bin_path, prototxt=prototxt, caffemodel=caffemodel, binproto=binproto, words=words, img=filename)
    print('trying command:'+str(detect_command))
    retvals = commands.getstatusoutput(detect_command)
    logging.debug('return from command ' + detect_command + ':' + str(retvals), end="\n")
    return retvals

#run a parallelization test
if __name__ == "__main__":
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
        cats_dict = caffe_general(url, async = True)
        print(str(cats_dict))
    elapsed_time = time.time() - start_time
    print('tot elapsed:'+str(elapsed_time)+',per image:'+str(float(elapsed_time)/len(urls)))


