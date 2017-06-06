__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
"""
run this like:
gunicorn -b :8080 -w 1 -k gevent -n nd --timeout 120 trendi.paperdoll.neurodoll_falcon:api
assuming the docker was started with port 8084 specified e.g.
nvidia-docker run -it -v /data:/data -p 8080:8080 --name nd eu.gcr.io/test-paper-doll/tg/base_all_machine_learning:2
"""


import traceback
import falcon
from jaweson import json, msgpack

from .. import neurodoll
    #, neurodoll_single_category
#from .. import neurodoll_with_multilabel
from .. import constants
# from .darknet.pyDarknet import mydet
from trendi import Utils


print "Done with imports"

class NeurodollResource:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp):
        """Handles GET requests"""
        print "Reached on_get, send to post"
        self.on_post(req,resp)
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }
        print('neurodollresource got get request')
        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        print "Reached on_post"
        category_index = req.get_param('categoryIndex')
        if category_index:
            print('got req for category index '+str(category_index))
            category_index = int(category_index)

        threshold = req.get_param('threshold')
        if threshold:
            print('got threshold '+str(threshold))

        get_multilabel_results = req.get_param('getMultilabelResults')
        if get_multilabel_results:
            print('got req for multi:'+str(get_multilabel_results))
            get_multilabel_results = get_multilabel_results in ["true", "True", True]

        get_combined_results = req.get_param('getCombinedResults')
        if get_combined_results in ["true", "True", True]:
            print('got req for  combined:'+str(get_combined_results))
            get_combined_results = True

        get_layer_output = req.get_param('getLayerOutput')
        if get_layer_output:
            print('got req for layer output:'+str(get_layer_output))

        get_all_graylevels = req.get_param('getAllGrayLevels')
        if get_all_graylevels in ["true", "True", True]:
            print('got req for all graylevels:'+str(get_all_graylevels))
            get_all_graylevels = True

        get_category_graylevel = req.get_param('getCategoryGraylevel')
        if get_category_graylevel:
            print('got req for graylevel:'+str(get_category_graylevel))


        posted_url = req.get_param('imageUrl')
        if posted_url:
            print('got url:'+posted_url)

        ret = {"success": False}

        try:
            if posted_url:
                img = Utils.get_cv2_img_array(posted_url)
            else:
                data = msgpack.loads(req.stream.read())
                img = data.get("image")
            print('got img of type:'+str(type(img)))
#            if get_yolo_results:
#                yolo_output = mydet.get_yolo_results(img)
#                ret['yolo_output'] = yolo_output
#                print('yolo output:'+str(yolo_output))

        #all graylevel outputs
            if get_all_graylevels:
                all_graylevel_output = neurodoll.get_all_category_graylevels(img)
                ret['all_graylevel_output'] = all_graylevel_output
                if all_graylevel_output is not None:
                    print('all graylevel output shape:'+str(all_graylevel_output.shape))
                    ret["success"] = True


        # multilabel alone
            if get_multilabel_results:
                multilabel_output = neurodoll.get_multilabel_output(img)
                ret['multilabel_output'] = multilabel_output
                print('multilabel output:'+str(multilabel_output))
                if multilabel_output is not None:
                    ret["success"] = True
                # ret["success"] = bool(multilabel_output)

        # combined multilabel and nd
            if get_combined_results:

                #for old nd with all cats
#                combined_output = neurodoll.combine_neurodoll_and_multilabel(img)
                #for new nd with 'v3 cats' - upper cover , fullbody etc.
                try:
                    combined_output = None
                    combined_output = neurodoll.combine_neurodoll_v3labels_and_multilabel(img)
                    ret['combined_output'] = combined_output
                    ret['mask'] = combined_output
                except:
                    print('error in combine_neurodoll_v3labels_and_multilabel_using_graylevel ')
                if combined_output is not None:
                    ret["success"] = True

        # yonti style - single category mask
            ret["label_dict"] = constants.ultimate_21_dict

            if category_index:
                if threshold:
                    print('neurodoll falcon sending img and threshold to get_cat_gl_masked_thresholded')
                    ret["mask"] = neurodoll.get_category_graylevel_masked_thresholded(img, category_index,threshold=threshold)
                else:
                    print('neurodoll falcon sending img without threshold to get_cat_gl_masked_thresholded')
                    ret["mask"] = neurodoll.get_category_graylevel_masked_thresholded(img, category_index)
                if ret["mask"] is not None:
                    ret["success"] = True

        # layer output for yonti - default is last fc layer (myfc7) but any can be accessed (put layer name as argument)
            if get_layer_output:
                ret["layer_output"] = neurodoll.get_layer_output(img,layer=get_layer_output)

#                url_or_np_array,required_image_size=(256,256),layer='myfc7'


                if ret["layer_output"] is not None:
                    ret["success"] = True
                else:
                    ret["error"] = "no layer output obtained"

        # regular neurodoll call
            if not get_multilabel_results and not get_combined_results and not category_index:
                print "No special params, inferring..."
                ret["mask"],labels = neurodoll.infer_one(img)
                if ret["mask"] is not None:
                    ret["success"] = True
                else:
                    ret["error"] = "No mask from ND"


        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()
            url = req.get_param('image')
            ret['url'] = url
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/nd/', NeurodollResource())
