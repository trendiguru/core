import traceback
import falcon
import os
#this file has to go in the rcnn folder
import rcnn_demo
import requests

from jaweson import json, msgpack

print('falcon is coming form '+str(falcon.__file__))
base_dir = os.path.dirname(os.path.realpath(__file__))
print('current_dir is '+str(base_dir))

print "Done with imports"

FRCNN_CLASSIFIER_ADDRESS = "http://13.82.136.127:8082/frcnn"
#what is the frcnn referring to

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
        print "Reached on_post"
        gpu = req.get_param('gpu')
        ret = {"success": False}
#
        try:
#            data = msgpack.loads(req.stream.read())
            data = req.stream.read()
            print('data:'+str(data))
#            img = data.get("image")
#            img = data['name']
            img = data.split('"')[1]
            print('img:'+str(img))
            fcrnn_output = self.get_fcrnn_output(img)
            output = rcnn_demo.get_rcnn_output(img)
            ret["output"] = output
            if ret["output"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No output from rcnn"
#            self.write_log(url,output)

        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200

    def write_log(self,url,output):
        with open('/data/jeremy/caffenets/hydra/production/hydra/logged_output.txt','a') as fp:
            output['url']=url
            json.dump(output,fp,indent=4)
            fp.write()

    def get_fcrnn_output(self,url):
        data = msgpack.dumps({"image": url})
        params = {}
        resp = requests.post(FRCNN_CLASSIFIER_ADDRESS, data=data, params=params)
        print('response from fcrnn:'+str(resp.content))
        return (resp.content)


api = falcon.API()
#api.add_route('/mlb3/', HydraResource())
api.add_route('/hydra/', HydraResource())
#send post request to http://ip-of-server/hydra
#make sure to run docker container with -p