__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
import traceback
import falcon
# from .darknet.pyDarknet import mydet

from jaweson import json, msgpack
import os
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
            with open(filename, 'wb') as f:
                f.write(imagedata)
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


def gen_json(images_dir='data/pd_output',annotations_dir='data/pd_output',
             outfile = 'data/pd_output.json',labels=constants.pixlevel_categories_v2,mask_suffix='_pixv2_webtool.png',
             ignore_finished=True,finished_mask_suffix='_pixv2_webtool_finished_mask.png'):
    images = [os.path.join(images_dir,f) for f in os.listdir(images_dir) if '.jpg' in f]
    the_dict = {'labels': labels, 'imageURLs':[], 'annotationURLs':[]}

    for f in images:
        annotation_file = os.path.basename(f).replace('.jpg',mask_suffix)
        annotation_file = os.path.join(annotations_dir,annotation_file)
        if not os.path.isfile(annotation_file):
            print('could not find '+str(annotation_file))
            continue
        if ignore_finished:
            maskname = annotation_file.replace(mask_suffix,finished_mask_suffix)
            print('maskname:'+maskname)
            if os.path.isfile(maskname):
                print('mask exists, skipping')
                continue
        the_dict['imageURLs'].append(f)
        the_dict['annotationURLs'].append(annotation_file)
        print('added image '+f+' mask '+annotation_file)
    with open(outfile,'w') as fp:
        json.dump(the_dict,fp,indent=4)


api = falcon.API()
api.add_route('/pixlevel_annotator/', PixlevelResource())
