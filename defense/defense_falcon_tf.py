__author__ = 'jeremy'

"""
run this like:
gunicorn -b :8083 -w 1 -k gevent -n fervent_torvalds --timeout 120 trendi.defense.defense_falcon_tf:api
assuming the docker was started with port 8072 specified e.g.
nvidia-docker run -it -v /data:/data -p 8083:8083 --name fervent_torvalds eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
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
import random
import string
import base64

from trendi.classifier_stuff.tf import tf_detect
from trendi import constants

print "Defense_falcon_tf done with imports"

# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
TF_HLS_CLASSIFIER_ADDRESS = constants.TF_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"

class HLS_TF:
    def __init__(self):
        print "Loaded Resource for TF"


    def on_get(self, req, resp): #/
        """Handles GET requests"""
        serializer = json
        resp.content_type = "application/json"
        print('\nStarting HLS_TF (got a get request)')
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
                print('db1')
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                if img_arr is None:
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
            print('db2')
            if r_x1 or r_x2 or r_y1 or r_y2:
                print('cropping image')
                img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 2:(", traceback.format_exc())
        try:
            print('db3')
            #which net to use - pyyolo or shell yolo , default to pyyolo
            imgpath = '/data/jeremy/tensorflow/tmp.jpg'
            tmpfile = cv2.imwrite(imgpath,img_arr)
            detected = tf_detect.analyze_image(imgpath,thresh=0.1)
            print('detected:'+str(detected))
            if (r_x1, r_y1) != (0, 0):
                print('moving bbs due to crop')
                for obj in detected:
                    try:
                        x1, y1, x2, y2 = obj["bbox"]
                        obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
                    except (KeyError, TypeError):
                        print "No valid 'bbox' in detected"
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 3:(", traceback.format_exc())
        try:
            print('db4')
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong in get section 4:(", traceback.format_exc())


    def on_post(self, req, res):
        #untested
        print('\nStarting combine_gunicorn tf hls (got a post request)')
        start_time=time.time()
        tmpfile = '/data/jeremy/image_dbs/variant/viettel_demo/'
        N=10
        randstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
        tmpfile = os.path.join(tmpfile,randstring+'.jpg')

        sent_timestamp=0
 #      serializer = msgpack
 #      res.content_type = "application/x-msgpack"
        try:
            json_data = json.loads(req.stream.read().decode('utf8'))
            if 'sent_timestamp' in json_data:
                sent_timestamp = float(json_data['sent_timestamp'])
                print('sent timestamp {}'.format(sent_timestamp))
            else:
                sent_timestamp=0 #
            xfer_time = time.time()-sent_timestamp
            print('xfer time:{}'.format(xfer_time))
            base64encoded_image = json_data['image_data']
            data = base64.b64decode(base64encoded_image)
#            print('data type {}'.format(type(data)))
            with open(tmpfile, "wb") as fh:
                fh.write(data)
                print('wrote file to {}, elapsed time for xfer {}'.format(tmpfile,time.time()-start_time))
                decode_time = time.time()-start_time
            try:
                print('db2')
#                imgpath = '/data/jeremy/tensorflow/tmp.jpg'
#                tmpfile = cv2.imwrite(imgpath,img_arr)
                detected = tf_detect.analyze_image(tmpfile,thresh=0.2)

            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 3:(", traceback.format_exc())
            try:
                print('db4')
                res.data = json.dumps(detected)
                res.status = falcon.HTTP_200
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 4:(", traceback.format_exc())
            try:
                self.write_log('id',detected)
            except:
                raise falcon.HTTPBadRequest("Something went wrong in get section 5 (wrte log):(", traceback.format_exc())

#             stream = req.stream.read()
#             print('stream {}'.format(stream))
#             data = serializer.loads(stream)
#             print('data:{}'.format(data))
#             img_arr = data.get("image")
#             print('img arr shape {}'.format(img_arr.shape))
# #            detected = self.detect_yolo_pyyolo(img_arr)
#             cv2.imwrite(tmpfile,img_arr)
#             detected = self.tracker.next_frame(tmpfile)
#             resp.data = serializer.dumps({"data": detected})
#             resp.status = falcon.HTTP_200

        except:
            raise falcon.HTTPBadRequest("Something went wrong in post :(", traceback.format_exc())

        res.status = falcon.HTTP_203
        res.body = json.dumps({'status': 1, 'message': 'success','data':json.dumps(detected)})
#        res.body = json.dumps(detected)


#            detected = self.detect_yolo_pyyolo(img_arr)




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

