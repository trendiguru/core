"""
falcon for hydra , run this like:
gunicorn -b :8081 -w 1 -k gevent -n hls --timeout 120 trendi.defense.defense_falcon:api
assuming the docker was started with port 8081 specified e.g.
nvidia-docker run -it -v /data:/data -p 8081:8081 --name hls_hydra  eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2 sh -c 'git -C /usr/lib/python2.7/dist-packages/trendi pull && /bin/bash'
"""

import traceback
import falcon
print(falcon.__file__)
#from .. import multilabel_from_hydra_hls
from .. import multilabel_from_hydra_tg
import requests
from .. import Utils
import numpy as np

from jaweson import json, msgpack

print "Done with imports"


class HydraResource:
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
        print "Reached hydra on_post"
        gpu = req.get_param('gpu')
        ret = {"success": False}
#
        try:
#            data = msgpack.loads(req.stream.read())
            data = req.stream.read()
#            print('data coming into hydra:'+str(data))
            print('hydra falcon')
            dict = json.loads(data)
            img = dict.get("image")

#            img = data['name']
#            img = data.split('"')[1]
  #          img = data
            if isinstance(img,basestring):
                print('url coming to hydra falcon:'+str(img))
            else:
                print('img arr into hydra falcon size:'+str(img.shape))
 #           img_arr=Utils.get_cv2_img_array(img)
#            frcnn_output =  self.get_fcrnn_output(self,img)
#            hydra_output = multilabel_from_hydra_hls.get_hydra_output(img,detection_threshold=0.9)
            hydra_output = multilabel_from_hydra_tg.get_hydra_output(img)
            if "sweater_binary_h_iter_50000" in hydra_output:
                del hydra_output["sweater_binary_h_iter_50000"]
            if "sweatshirt_binary_h_iter_14000" in hydra_output:
                del hydra_output["sweatshirt_binary_h_iter_14000"]
            if "backpack_hydra_iter_2000" in hydra_output:
                del hydra_output["backpack_hydra_iter_2000"]


            del hydra_output["url"] #dont need this , its an array anyway lately

            ret["output"] = hydra_output
            if ret["output"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No output from mlb"
#            self.write_log(url,output)

        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()

#        resp.data = msgpack.dumps(ret)
#        resp.content_type = 'application/x-msgpack'
        resp.data = json.dumps(ret) #maybe dump instead of dumps?
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200

    def write_log(self,url,output):
        with open('/data/jeremy/caffenets/hydra/production/hydra/logged_output.txt','a') as fp:
            output['url']=url
            json.dump(output,fp,indent=4)
            fp.write()




api = falcon.API()
#api.add_route('/mlb3/', HydraResource())
api.add_route('/hydra/', HydraResource())
#send post request to http://ip-of-server/hydra
#make sure to run docker container with -p