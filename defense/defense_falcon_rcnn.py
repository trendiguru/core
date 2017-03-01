"""
run this like:
gunicorn -b :8082 -w 1 -k gevent -n hls --timeout 120 trendi.defense.defense_falcon_rcnn:api
assuming the docker was started with port 8082 specified e.g.
nvidia-docker run -it -v /data:/data -p 8082:8082 --name frcnn eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""
import traceback
import falcon
import os
import cv2
import numpy as np
#this file has to go in the rcnn folder
import defense_rcnn
import requests

from jaweson import json #, msgpack

from trendi import constants

# print('falcon is coming form '+str(falcon.__file__))
# base_dir = os.path.dirname(os.path.realpath(__file__))
# print('current_dir is '+str(base_dir))

print "Done with imports"

HYDRA_CLASSIFIER_ADDRESS = constants.HYDRA_HLS_CLASSIFIER_ADDRESS #"http://13.82.136.127:8081/hydra"
FRCNN_CLASSIFIER_ADDRESS = constants.FRCNN_CLASSIFIER_ADDRESS #"http://13.82.136.127:8082/hls"
#what is the frcnn referring to - maybe its the thing at the end of file
#namely, api.add_route('/frcnn/', HydraResource())

class HLS:
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
            print('get request:'+str(req))
            raise falcon.HTTPMissingParam("imageUrl")
        else:
            try:
                response = requests.get(image_url)
                img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
                detected = self.detect(img_arr,url=image_url)
                resp.data = serializer.dumps({"data": detected})
                resp.status = falcon.HTTP_200
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


    def detect(self, img_arr,url=''):
        detected = defense_rcnn.detect_frcnn(img_arr)
        # get hydra results
        print('started defense_falcon_rcnn.detect')
        for item in detected:
            cat = item["object"]
            if cat == "person":
                print('bbox:'+str(item['bbox'])+' type:'+str(type(item['bbox'])))
                x1,y1,x2,y2 = item["bbox"]
                print('x1 {} y1 {} x2 {} y2 {} type {}:'.format(x1,y1,x2,y2,type(x1)))
                print('img arr type:'+str(type(img_arr)))
                print('img arr shape:'+str((img_arr.shape)))
                cropped_image = img_arr[y1:y2,x1:x2]
                # print('crop:{} {}'.format(item["bbox"],cropped_image.shape))
                hydra_output = self.get_hydra_output(cropped_image)
                if hydra_output:
                    item['details'] = hydra_output
        self.write_log(url,detected)
        return detected


    def get_hydra_output(self, subimage):
        '''
        get hydra details on an image
        :param subimage: np array , e..g a crop of the original which fcrnn has found
        :return:
        '''
        data = json.dumps({"image": subimage})
        print('defense falcon is attempting to get response from hydra at '+str(HYDRA_CLASSIFIER_ADDRESS))
        resp = requests.post(HYDRA_CLASSIFIER_ADDRESS, data=data)
        # print('resp:'+str(resp))
        # print('type;'+str(type(resp)))
        # print('resp:'+str(resp.content))
        # print('type;'+str(type(resp.content)))
        dict = json.loads(resp.content)
        # print('response dict from hydra:'+str(dict))
        return dict['output']


    def write_log(self, url, output):
        with open('/data/jeremy/caffenets/hydra/production/hydra/logged_hls_output.txt', 'a') as fp:
            output.append = {'url':url}
            json.dumps(output, fp, indent=4)
            fp.write()

api = falcon.API()
api.add_route('/hls/', HLS())