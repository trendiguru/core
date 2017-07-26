__author__ = 'jeremy'

import logging
import msgpack
import requests

from trendi import constants
logging.basicConfig(level=logging.INFO)
import pandas as pd
import hashlib
import time
import numpy as np
import cv2
import os
import sys
import json

from trendi import constants
from trendi.utils import imutils
from trendi import Utils

run_local_yolo=True
if run_local_yolo:
    import pyyolo
    print "bb_results done with imports"
    #run 14 aka v2
    datacfg = '/data/jeremy/darknet/cfg/hls_v2.data'
    cfgfile = '/data/jeremy/darknet/cfg/yolo.2.0.cfg' #higher res for smaller objects, increase subdivisions *4 to preserve memoery
    weightfile = '/data/jeremy/darknet/v2/yolo_120000_run14.weights'
    darknet_path = '/data/jeremy/pyyolo_tg/pyyolo'
    thresh = 0.1
    hier_thresh = 0.5
    new_pyyolo=True
    if new_pyyolo:
        pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)
    else:
        pyyolo.init(datacfg, cfgfile, weightfile)


# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
YOLO_HLS_CLASSIFIER_ADDRESS = constants.YOLO_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"
HYDRA_CLASSIFIER_ADDRESS = "http://hls_hydra:8081/hydra" # constants.HYDRA_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8081/hydra"



def bb_output_using_gunicorn(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url_or_np_array, get_multilabel_results=True)
    logging.debug('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        logging.warning('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    logging.debug('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def bb_output_yolo_using_api(url_or_np_array,CLASSIFIER_ADDRESS=constants.YOLO_HLS_CLASSIFIER_ADDRESS,roi=None,get_or_post='GET',query='file'):
    logging.debug('starting bb_output_api at addr '+str(CLASSIFIER_ADDRESS))
#    CLASSIFIER_ADDRESS =   # "http://13.82.136.127:8082/hls"
    if isinstance(url_or_np_array,basestring): #got a url (use query= 'imageUrl') or filename, use query='file' )
        data = {query: url_or_np_array}
        logging.debug('using imageUrl as data')
    else:
        img_arr = url_or_np_array
        jsonified = pd.Series(img_arr).to_json(orient='values')

        data = {"image": jsonified} #this was hitting 'cant serialize' error
        logging.debug('using image as data')
    if roi:
        logging.debug("Make sure roi is a list in this order [x1, y1, x2, y2]")
        data["roi"] = roi
#    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    if get_or_post=='GET':
        result = requests.get(CLASSIFIER_ADDRESS,params=data)
    else:
        serialized_data = msgpack.dumps(data)
  #      result = requests.post(CLASSIFIER_ADDRESS,data=serialized_data)
        result = requests.post(CLASSIFIER_ADDRESS,data=data)

    if result.status_code is not 200:
       print("Code is not 200")
#     else:
#         for chunk in result.iter_content():
#             print(chunk)
# #            joke = requests.get(JOKE_URL).json()["value"]["joke"]

#    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    c = result.content
    #content should be roughly in form
#    {"data":
    # [{"confidence": 0.366, "object": "car", "bbox": [394, 49, 486, 82]},
    # {"confidence": 0.2606, "object": "car", "bbox": [0, 116, 571, 462]}, ... ]}
    if not 'data' in c:
        print('didnt get data in result from {} on sendng {}'.format(CLASSIFIER_ADDRESS,data))
    return eval(c) # c is a string, eval(c) is dict

def detect_hls(img_arr, roi=[],CLASSIFIER_ADDRESS=constants.YOLO_HLS_CLASSIFIER_ADDRESS):
    print('using addr '+str(CLASSIFIER_ADDRESS))
    data = {"image": img_arr}
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serializer = json
    serialized_data = serializer.dumps(data)
#    serialized_data = msgpack.dumps(data)
#    resp = requests.post(YOLO_HLS_ADDRESS, data=data)
    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    print('resp from hls:'+str(resp))
    print('respcont from hls:'+str(resp.content))
    print('respctest from hls:'+str(resp.text))
    return msgpack.loads(resp.content)

def local_yolo(img_arr, url='',classes=constants.hls_yolo_categories,save_results=True):
#                item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':'>'+str(thresh)}
    print('started local pyyolo detect, thresh='+str(thresh)+' hier '+str(hier_thresh))
    save_path = '/data/jeremy/pyyolo/results/'
    if img_arr is None:
        print('got None img array!!')
        None
    if len(img_arr.shape) == 2: #got 1-chan(gray) image
        print('got gray img')
        img_arr_bgr=np.zeros([img_arr.shape[0],img_arr.shape[1],3])
        img_arr_bgr[:,:]=img_arr #copy bw image into all channels
        print('sizes: {} {}'.format(img_arr_bgr,img_arr))
        img_arr = img_arr_bgr
    print('img arr size {}'.format(img_arr.shape))
    max_image_size=1200 #1280 fails
    resized=False
    if img_arr.shape[0]>max_image_size or img_arr.shape[1]>max_image_size:  #maybe this is causing prob at http://a.abcnews.com/images/US/150815_yt_phillypd_33x16_1600.jpg
        maxside=max(img_arr.shape)
        reduction_factor = maxside/1000.0 #force maxside to 1000
        original_size=img_arr.shape
        resized=True
        img_arr = cv2.resize(img_arr,(int(img_arr.shape[1]/reduction_factor),int(img_arr.shape[0]/reduction_factor)))
        print('reduced size to {}'.format(img_arr.shape))
        #generate randonm filename
    hash = hashlib.sha1()
    hash.update(str(time.time()))
    if save_results:
        timestr = time.strftime("%Y%m%d.%H%M%S")
        img_filename = timestr+hash.hexdigest()[:5]+'.jpg'
        Utils.ensure_dir(save_path)
        img_path = os.path.join(save_path,img_filename)
        print('detect_yolo_pyyolo saving file '+str(img_path))
        try:
            cv2.imwrite(img_path,img_arr)
        except:
            print('some trouble saving image,'+str(sys.exc_info()[0]))
    relevant_items = []
    print('getting results from get_pyyolo_results')
    yolo_results = get_local_pyyolo_results(img_arr)
    print('got results from get_pyyolo_results')

    for item in yolo_results:
        print(item)
        if resized:
            img_arr = cv2.resize(img_arr,(original_size[1],original_size[0]))
            item['bbox'][0]=int(item['bbox'][0]*reduction_factor)
            item['bbox'][1]=int(item['bbox'][1]*reduction_factor)
            item['bbox'][2]=int(item['bbox'][2]*reduction_factor)
            item['bbox'][3]=int(item['bbox'][3]*reduction_factor)
            print('fixed size back to original {}'.format(item))

        xmin=item['bbox'][0]
        ymin=item['bbox'][1]
        xmax=item['bbox'][2]
        ymax=item['bbox'][3]
        assert xmin<xmax,'xmin not < xmax!!!'
        assert ymin<ymax,'xmin not < xmax!!!'

        relevant_items.append(item)

    if save_results:
        imutils.bb_with_text(img_arr,[xmin,ymin,(xmax-xmin),(ymax-ymin)],item['object'])
        marked_imgname = img_path.replace('.jpg','_bb_yolos.jpg')
        json_name = img_path.replace('.jpg','.json')
        print('pyyolo bb image being writtten to '+str(marked_imgname))
        print('pyyolo bb data being writtten to '+str(json_name))
        try:
            with open(json_name,'w') as fp:
                json.dump(yolo_results,fp,indent=4)
            r=cv2.imwrite(marked_imgname,img_arr)
          #skip saving bb image to conserve space
            print('imgwrite result '+str(r))
        except:
            print('some trouble saving bb image or data:'+str(sys.exc_info()[0]))

    print('detect yolo returning:'+str(relevant_items))
    return relevant_items


def get_local_pyyolo_results(img_arr, url='', classes=constants.hls_yolo_categories, method='file'):
    # from file
    relevant_bboxes = []
    if method == 'file':
        print('----- test original C using a file')
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        img_filename = hash.hexdigest()[:10]+'pyyolo.jpg'
      #  img_filename = 'incoming.jpg'
        cv2.imwrite(img_filename,img_arr)
        outputs = pyyolo.test(img_filename, thresh, hier_thresh)

#not sure what the diff is between this second method (pyyolo.detect) and first (pyyolo.test)
            #except it uses array instead of file
# print('----- test python API using a file')
    else:
        i = 1  #wtf is this count for
        while i < 2:
            # ret_val, img = cam.read()
#                img = cv2.imread(filename)
            img = img_arr.transpose(2,0,1)
            c, h, w = img.shape[0], img.shape[1], img.shape[2]
            # print w, h, c
            data = img.ravel()/255.0
            data = np.ascontiguousarray(data, dtype=np.float32)
            print('calling pyyolo.detect')
            try:
                outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
            except:
                print('some trouble calling pyyolo detect,'+str(sys.exc_info()[0]))

            print('returned from  pyyolo.detect')
            for output in outputs:
                print(output)
            i = i + 1
# free model
    for output in outputs:
        print(output)
        label = output['class']
        if 'person' in label:
            label='person'  #convert 'person_wearing_red/blue_shirt' into just person
        xmin = output['left']
        ymin = output['top']
        xmax = output['right']
        ymax = output['bottom']
        conf = output['prob']
        item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(conf,4)}
        h,w=img_arr.shape[0:2]
        frac = 5
        cropped_arr = img_arr[h/frac:h-(h/frac),w/frac:w-(w/frac)]
        dominant_color = imutils.dominant_colors(cropped_arr)
        print('dominant color:'+str(dominant_color))
        if dominant_color is not None:
            item['details']={'color':dominant_color}
#            item = {'object':label,'bbox':[xmin,ymin,xmax,ymax],'confidence':round(float(confidence),3)}
        relevant_bboxes.append(item)

#        pyyolo.cleanup()
    return relevant_bboxes

