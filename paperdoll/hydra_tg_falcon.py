"""
run this like:
gunicorn -b :8083 -w 1 -k gevent -n hls --timeout 120 trendi.paperdoll.hydra_falcon:api
assuming the docker was started with port 8083 specified e.g.
nvidia-docker run -it -v /data:/data -p 8083:8083 --name hydra_tg eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""
#from __future__ import absolute_import
import traceback
import os
import falcon
import sys

# print('falcon is coming form '+str(falcon.__file__))
# base_dir = os.path.dirname(os.path.realpath(__file__))
# print('current_dir is '+str(base_dir))
import cv2
import numpy as np
#this file has to go in the rcnn folder
import requests
from jaweson import json #, msgpack

from trendi import multilabel_from_hydra_tg

# print('falcon is coming form '+str(falcon.__file__))
# base_dir = os.path.dirname(os.path.realpath(__file__))
# print('current_dir is '+str(base_dir))

print "Done with imports"

#
class HYDRA_TG:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp): #
        """Handles GET requests"""
        # if req.client_accepts_msgpack or "msgpack" in req.content_type:
        #     serializer = msgpack
        #     resp.content_type = "application/x-msgpack"
        # else:
        serializer = json
        resp.content_type = "application/json"

        image_url = req.get_param("imageUrl")
        if not image_url:
            print('get request didnt specify a url:'+str(req))
            raise falcon.HTTPMissingParam("imageUrl")
        else:
            print('hydra_tg falcon on_get got url '+image_url)
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                detected = self.detect(img_arr)
                self.write_log(image_url,detected)
                resp.data = serializer.dumps({"data": detected})
                resp.status = falcon.HTTP_200
                self.write_log(image_url,detected)
            except:
                raise falcon.HTTPBadRequest("Something went wrong :(", traceback.format_exc())


    def on_post(self, req, resp):
        # if req.client_accepts_msgpack or "msgpack" in req.content_type:
        #     serializer = msgpack
        #     resp.content_type = "application/x-msgpack"
        # else:
        serializer = json
        resp.content_type = "application/json"
        try:
            data = serializer.loads(req.stream.read())
            img_arr = data.get("image")
            detected = self.detect(img_arr)
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong :(", traceback.format_exc())


    def detect(self, img_arr):
        print('started hydra_falcon.detect')
        try:
            detected = multilabel_from_hydra_tg.get_hydra_output(img_arr)
        # get hydra results
        except:
            print("Unexpected error in hydra_falcon.detect:"+ str(sys.exc_info()[0]))
            return None
        return detected


    def write_log(self, url, output):
        with open('/data/jeremy/caffenets/hydra/production/hydra/tg_logged_output.txt', 'a+') as fp: #a+ not a to create if file doesnt exist
            out=output
            out['url']=url
            json.dumps(out, fp, indent=4)
#            fp.write()
#
api = falcon.API()
api.add_route('/hydra_tg/', HYDRA_TG())