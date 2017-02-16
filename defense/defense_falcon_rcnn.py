#run this like:
#gunicorn -b :8082 -w 1 -k gevent -n hls --timeout 120 trendi.defense.defense_falcon_rcnn:api
#assuming the docker was started with port 8082 specified e.g.
#nvidia-docker run -it -v /data:/data -p 8082:8082 --name frcnn eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'

import traceback
import falcon
import os
import cv2
import numpy as np
#this file has to go in the rcnn folder
import defense_rcnn
import requests

from jaweson import json, msgpack

print('falcon is coming form '+str(falcon.__file__))
base_dir = os.path.dirname(os.path.realpath(__file__))
print('current_dir is '+str(base_dir))

print "Done with imports"

HYDRA_CLASSIFIER_ADDRESS = "http://13.82.136.127:8081/hydra"
FRCNN_CLASSIFIER_ADDRESS = "http://13.82.136.127:8082/frcnn"
#what is the frcnn referring to - maybe its the thing at the end of file
#namely, api.add_route('/frcnn/', HydraResource())

class FrcnnResource:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp): #
        """Handles GET requests"""
        quote = {
            'msg': 'responding to posts... ',
            'author': 'jeremy rutman'
        }
        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        print "Reached on_post"
        gpu = req.get_param('gpu')
        ret = {"success": False}
        ret_hydra = {"success":False}
    #
        try:
#            data = msgpack.loads(req.stream.read())
            data = req.stream.read()
            print('data coming into frcnn:'+str(data))
#            img = data.get("image")
#            img = data['name']
            url = data.split('"')[1]
            print('url:'+str(url))
            response = requests.get(url)  # download
            img_arr = cv2.imdecode(np.asarray(bytearray(response.content)), 1)


            output = defense_rcnn.detect_frcnn(img_arr)
            print('frcnn output:'+str(output))
            ret["output"] = output
            if ret["output"] is not None:
                ret["success"] = True
            else:
                ret["error_frcnn"] = "No output from rcnn"
            print('done with frcnn')
#            self.write_log(url,output)

        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()


    #get hydra results
        print('done with frcnn now doing hydra')
        try:
            for item in output:
                print('item:'+str(item))
                cat = item["object"]
                print('category:'+str(cat))
                if cat == "person":
                if 1:
                    x1,y1,x2,y2 = item["bbox"] #these are x1y1x2y2
                    cropped_image = img_arr[y1:y2,x1:x2]
                    print('crop:{} {}'.format(item["bbox"],cropped_image.shape()))
                    hydra_output = self.get_hydra_output(cropped_image)
        except Exception as e:
            print('exception calling hydra {}'.format(e))
            traceback.print_exc()
            hydra_output["error_hydra"] = traceback.format_exc()
#        all_output = output.copy()
#        all_output.update(hydra_output)
#        print('total out:'+str(all_output))
        ret["output"] = output

#        resp.data = msgpack.dumps(ret)
#        resp.content_type = 'application/x-msgpack'
        resp.data = json.dumps(ret)
#        resp.content_type = 'text/plain'
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200

    def write_log(self,url,output):
        with open('/data/jeremy/caffenets/hydra/production/hydra/logged_output.txt','a') as fp:
            output['url']=url
            json.dump(output,fp,indent=4)
            fp.write()


    def get_hydra_output(self,url):
            #should be changed to sending img. array
#        data = msgpack.dumps({"image": url})
        data = url
        params = {}
        print('defense falcon is attempting to get response from hydra at '+str(HYDRA_CLASSIFIER_ADDRESS))
        resp = requests.post(HYDRA_CLASSIFIER_ADDRESS, data=data, params=params)
        print('response from hydra:'+str(resp.content))
        return resp.content

api = falcon.API()
#api.add_route('/mlb3/', HydraResource())
api.add_route('/frcnn/', FrcnnResource())
#send post request to http://ip-of-server/hydra
#make sure to run docker container with -p