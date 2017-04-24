"""
run this like:
gunicorn -b :8082 -w 1 -k gevent -n hls --timeout 120 trendi.defense.defense_falcon_rcnn:api
assuming the docker was started with port 8082 specified e.g.
nvidia-docker run -it -v /data:/data -p 8082:8082 --name frcnn eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""
import traceback
import falcon
from falcon_cors import CORS
import os
import cv2
import numpy as np
#this file has to go in the rcnn folder
import defense_rcnn
import requests

from jaweson import json, msgpack

from trendi import constants

# print('falcon is coming form '+str(falcon.__file__))
# base_dir = os.path.dirname(os.path.realpath(__file__))
# print('current_dir is '+str(base_dir))

print "Done with imports"

# Containers must be on the same docker network for this to work (otherwise go backt o commented IP address
HYDRA_CLASSIFIER_ADDRESS = "http://hls_hydra:8081/hydra" # constants.HYDRA_HLS_CLASSIFIER_ADDRESS # "http://13.82.136.127:8081/hydra"
FRCNN_CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS # "http://13.82.136.127:8082/hls"
# what is the frcnn referring to - maybe its the thing at the end of file
# namely, api.add_route('/frcnn/', HydraResource())

class HLS:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp): #
        """Handles GET requests"""
        serializer = json
        resp.content_type = "application/json"

        image_url = req.get_param("imageUrl")
        r_x1 = req.get_param_as_int("x1")
        r_x2 = req.get_param_as_int("x2")
        r_y1 = req.get_param_as_int("y1")
        r_y2 = req.get_param_as_int("y2")

        if not image_url:
            print('get request:' + str(req) + ' is missing imageUrl param')
            raise falcon.HTTPMissingParam("imageUrl")
        else:
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                if r_x1 or r_x2 or r_y1 or r_y2:
                    img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                    print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
                detected = self.detect(img_arr, url=image_url)
                if (r_x1, r_y1) != (0, 0):
                    for obj in detected:
                        try:
                            x1, y1, x2, y2 = obj["bbox"]
                            obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
                        except (KeyError, TypeError):
                            print "No valid 'bbox' in detected"
                            

                resp.data = serializer.dumps({"data": detected})
                resp.status = falcon.HTTP_200
            except:
                raise falcon.HTTPBadRequest("Something went wrong :(", traceback.format_exc())


    def on_post(self, req, resp):
        serializer = msgpack
        resp.content_type = "application/x-msgpack"
        try:
            data = serializer.loads(req.stream.read())
            img_arr = data.get("image")
            roi = data.get("roi")
            if roi:
                r_x1, r_y1, r_x2, r_y2 = roi
                img_arr = img_arr[r_y1:r_y2, r_x1:r_x2]
                print "ROI: {},{},{},{}; img_arr.shape: {}".format(r_x1, r_x2, r_y1, r_y2, str(img_arr.shape))
            detected = self.detect(img_arr)
            if roi and (r_x1, r_y1) != (0, 0):
                for obj in detected:
                    x1, y1, x2, y2 = obj["bbox"]
                    obj["bbox"] = x1 + r_x1, y1 + r_y1, x2 + r_x1, y2 + r_y1
            resp.data = serializer.dumps({"data": detected})
            resp.status = falcon.HTTP_200
        except:
            raise falcon.HTTPBadRequest("Something went wrong :(", traceback.format_exc())


    def detect(self, img_arr, url=''):
        detected = defense_rcnn.detect_frcnn(img_arr)
        print('started defense_falcon_rcnn.detect')
        for item in detected:
            cat = item["object"]
            if cat == "person":
                print('bbox:'+str(item['bbox'])+' type:'+str(type(item['bbox'])))
                x1,y1,x2,y2 = item["bbox"]
                print('x1 {} y1 {} x2 {} y2 {} type {}:'.format(x1,y1,x2,y2,type(x1)))
                print('img arr type:'+str(type(img_arr)))
                print('img arr shape:'+str((img_arr.shape)))
                cropped_image = img_arr[y1:y2, x1:x2]
                # print('crop:{} {}'.format(item["bbox"],cropped_image.shape))
                # get hydra results
                try:
                    hydra_output = self.get_hydra_output(cropped_image)
                    if hydra_output:
                        item['details'] = hydra_output
                except:
                    print "Hydra failed " + traceback.format_exc()
        self.write_log(url,detected)
        print detected
        return detected


    def get_hydra_output(self, subimage):
        '''
        get hydra details on an image
        :param subimage: np array , e..g a crop of the original which fcrnn has found
        :return:
        '''
        data = json.dumps({"image": subimage})
        print('defense falcon is attempting to get response from hydra at ' + str(HYDRA_CLASSIFIER_ADDRESS))
        try:
            resp = requests.post(HYDRA_CLASSIFIER_ADDRESS, data=data)
            dict = json.loads(resp.content)
            return dict['output']
        except:
            print('couldnt get hydra output')
            return None


    def write_log(self, url, output):
        logfile = '/data/jeremy/caffenets/hydra/production/hydra/logged_hls_output.txt'
        print('logging output to '+logfile)
        out = {'output':output,'url':url}
        with open(logfile, 'a') as fp:
           # output.append = {'url':url}
            json.dumps(out, fp, indent=4)
#            fp.write()


cors = CORS(allow_all_headers=True, allow_all_origins=True, allow_all_methods=True)
api = falcon.API(middleware=[cors.middleware])

api.add_route('/hls/', HLS())
