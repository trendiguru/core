import traceback
import falcon
print(falcon.__file__)
from .. import multilabel_from_hydra
import requests
from .. import Utils

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
            print('data coming into hydra:'+str(data))
#            img = data.get("image")
#            img = data['name']
#            img = data.split('"')[1]
            img = data
            print('url:'+str(img))
 #           img_arr=Utils.get_cv2_img_array(img)
#            frcnn_output =  self.get_fcrnn_output(self,img)
            output = multilabel_from_hydra.get_hydra_output(img,detection_threshold=0.9)
            if "sweater_binary_h_iter_50000" in output:
                del output["sweater_binary_h_iter_50000"]

            ret["output"] = output
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