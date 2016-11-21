__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
import traceback
import falcon
# from .darknet.pyDarknet import mydet

from jaweson import json, msgpack
import os
import subprocess

from trendi import constants

print "Done with imports"

class PixlevelResource:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'just work already ',
            'author': 'jeremy rutman'
        }
        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        print "Reached on_post"
        ret = {"success": False}

        try:
##            data = msgpack.loads(req.stream.read())
#            img = data.get("image")
            print('in try of onpost')
            data = json.loads(req.stream.read())
            print('data recd:'+str(data))

            filename = data["filename"]
            outfilename = filename.replace('.png','_finished_mask.png').replace('.bmp','_finished_mask.bmp')
            img_string = data["img_string"]
            imagedata = img_string.split(',')[-1].decode('base64')
            print('writing '+outfilename)
            with open(outfilename, 'wb') as f:
                f.write(imagedata)
            command_string = 'scp '+outfilename+' root@104.155.22.95:/var/www/js-segment-annotator/data/pd_output'
            subprocess.call(command_string, shell=True)

            #save mask under old name and send also
            with open(filename, 'wb') as f:
                f.write(imagedata)
            command_string = 'scp '+filename+' root@104.155.22.95:/var/www/js-segment-annotator/data/pd_output'
            subprocess.call(command_string, shell=True)

            ret["output"] = imagedata
            if ret["output"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No output from onpost"

        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200



api = falcon.API()
api.add_route('/pixlevel_annotator/', PixlevelResource())
