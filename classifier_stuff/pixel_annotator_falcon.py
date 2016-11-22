__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
import traceback
import falcon
# from .darknet.pyDarknet import mydet

from jaweson import json, msgpack
import os
import subprocess
import cv2

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
            img_string = data["img_string"]
            imagedata = img_string.split(',')[-1].decode('base64')

            #save new mask under old name and send
            with open(filename, 'wb') as f:
                f.write(imagedata)
                f.close()
            command_string = 'scp '+filename+' root@104.155.22.95:/var/www/js-segment-annotator/data/pd_output'
            subprocess.call(command_string, shell=True)

            #save new mask with 'finished_mask' filename and send
            #convert from 'webtool' format (index in red channel of 3chan img) to 'regular' format - 1 chan img that
            #cv2 reads in as 3chan with identical info in all chans
            img_arr = cv2.imread(filename)
            h,w = img_arr.shape[0:2]
            data = img_arr
            if len(img_arr.shape) == 3:
                data = img_arr[:,:,2]
            outfilename = filename.replace('.png','_finished_mask.png').replace('_webtool','').replace('.bmp','_finished_mask.bmp')
            print('writing rgb img to '+outfilename)
            cv2.imwrite(outfilename,data)
            command_string = 'scp '+outfilename+' root@104.155.22.95:/var/www/js-segment-annotator/data/pd_output'
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
