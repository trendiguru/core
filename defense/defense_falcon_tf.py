__author__ = 'jeremy'

"""
run this like:
gunicorn -b :8072 -w 1 -k gevent -n fervent_torvalds --timeout 120 trendi.defense.defense_falcon_tf:api
assuming the docker was started with port 8072 specified e.g.
nvidia-docker run -it -v /data:/data -p 8072:8072 --name fervent_torvalds eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""

import falcon
from falcon_cors import CORS

import traceback
import subprocess
import os
import requests
import hashlib
import time
from jaweson import json, msgpack
import numpy as np
import cv2
import sys
import codecs
import pandas as pd


from trendi.classifier_stuff.tf import tf_detect
from trendi import constants
from trendi import Utils
from trendi.utils import imutils

print "Defense_falcon_tf done with imports"

# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
TF_HLS_CLASSIFIER_ADDRESS = constants.TF_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"

class HLS_TF:
    def __init__(self):
        print "Loaded Resource for TF YOLO"


    def on_get(self, req, resp): #/
        """Handles GET requests"""
        serializer = json
        resp.content_type = "application/json"
        print('\nStarting HLS_YOLO (got a get request)')
        image_url = req.get_param("imageUrl")
        image = req.get_param("image")
        file = req.get_param("file")
        r_x1 = req.get_param_as_int("x1")
        r_x2 = req.get_param_as_int("x2")
        r_y1 = req.get_param_as_int("y1")
        r_y2 = req.get_param_as_int("y2")
        net = req.get_param("net")
        loc_thresh = req.get_param("threshold")
        loc_hier_thresh = req.get_param("hier_threshold")
#        for k,v in req.get_param.iteritems():
#            print('key {} value {}'.format(k,v))
        print('params into hls tf on_get: url {} file {} x1 {} x2 {} y1 {} y2 {} net {} thresh {} hierthresh {}'.format(image_url,file,r_x1,r_x2,r_y1,r_y2,net,loc_thresh,loc_hier_thresh))
        if loc_thresh is not None:
            global thresh
            thresh = float(loc_thresh)
        if loc_hier_thresh is not None:
            global hier_thresh
            hier_thresh = float(loc_hier_thresh)
        elif image_url:
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                if img_arr == None:
                    print('got none for image array')
                    resp.data = serializer.dumps({"data": 'bad image at '+image_url})
                    resp.status = falcon.HTTP_200
                    return
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 1:(", traceback.format_exc())
        elif image:
            print('getting img_arr directly')
            img_arr = pd.read_json(image,orient='values')
            print('img size {}'.format(img_arr.shape))
        elif file:
            print('getting file {}'.format(file))
            if not os.path.exists(file):
                raise falcon.HTTPBadRequest("could not get file "+str(file), traceback.format_exc())
            img_arr = cv2.imread(file)
            print('img size {}'.format(img_arr.shape))
        else:
            print('get request to hls tf:' + str(req) + ' is missing both imageUrl and image param')
            raise falcon.HTTPMissingParam("imageUrl,image")
        try:
            if r_x1 or r_x2 or r_y1 or r_y2:
                img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 2:(", traceback.format_exc())
        try:
            #which net to use - pyyolo or shell yolo , default to pyyolo
            imgpath = '/data/jeremy/tensorflow/tmp.jpg'
            tmpfile = cv2.imwrite(imgpath,img_arr)
            detected = tf_detect.analyze_image(imgpath)
            print('detected:'+str(detected))
            if (r_x1, r_y1) != (0, 0):
                for obj in detected:
                    try:
                        x1, y1, x2, y2 = obj["bbox"]
                        obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
                    except (KeyError, TypeError):
                        print "No valid 'bbox' in detected"
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 3:(", traceback.format_exc())
        try:
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 4:(", traceback.format_exc())

    def on_post(self, req, resp):
        #untested
        serializer = json
        resp.content_type = "application/x-msgpack"
        print('\nStarting HLS_YOLO (posted a post request)')
        try:
            data = serializer.loads(req.stream.read())
            print('data:{}'.format(data))
            img_arr = data.get("image")
            print('img arr shape {}'.format(img_arr.shape))
            roi = data.get("roi")
            if roi:
                r_x1, r_y1, r_x2, r_y2 = roi
                img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
            detected = self.detect_yolo_pyyolo(img_arr)
            if roi and (r_x1, r_y1) != (0, 0):
                for obj in detected:
                    x1, y1, x2, y2 = obj["bbox"]
                    obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong in post :(", traceback.format_exc())



    def write_log(self, url, output,filename='/data/jeremy/pyyolo/results/bbs.txt'):
#        logfile = '/data/jeremy/caffenets/hydra/production/hydra/logged_hls_output.txt'
        print('logging output to '+filename)
        out = {'output':output,'url':url}
        with open(filename, 'w+') as fp:
            print('writing :'+str(out))
           # output.append = {'url':url}
            json.dump(out, fp, indent=4)
            fp.close()
    #            fp.write()



cors = CORS(allow_all_headers=True, allow_all_origins=True, allow_all_methods=True)
api = falcon.API(middleware=[cors.middleware])

api.add_route('/hls/', HLS_TF())
# if __name__=="__main__":
#     img_arr = cv2.imread('/data/jeremy/image_dbs/bags_for_tags/photo_10006.jpg')
#     res = detect_yolo_pyyolo(img_arr)
#     print(res)

