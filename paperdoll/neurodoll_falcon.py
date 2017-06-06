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
        print "Reached on_get"
        
        ret = {"success": False}
        
        # Query Params
        category_index = req.get_param_as_int('categoryIndex')
        threshold = req.get_param('threshold')
        get_multilabel_results = req.get_param_as_bool('getMultilabelResults')
        get_combined_results = req.get_param_as_bool('getCombinedResults')
        get_layer_output = req.get_param_as_bool('getLayerOutput')
        get_all_graylevels = req.get_param_as_bool('getAllGrayLevels')
        get_category_graylevel = req.get_param_as_bool('getCategoryGraylevel')

        image_url = req.get_param('imageUrl', required=True)

        try:
            img = image_url # Utils.get_cv2_img_array(image_url)

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
                combined_output = None
                combined_output = neurodoll.combine_neurodoll_v3labels_and_multilabel(img)
                ret['combined_output'] = combined_output
                ret['mask'] = combined_output
                ret['show_results'] = True
                if combined_output is not None:
                    ret['success'] = True

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
        
        if ret["success"] and ret["show_results"]:
            raise falcon.HTTPFound("http://13.69.27.202:8099/")
        
        resp.data = json.dumps(ret)
        resp.content_type = 'application/json'
        resp.status = falcon.HTTP_200
        return(ret)

    def on_post(self, req, resp):
        print "Reached on_post"
        
        ret = {"success": False}
        
        # Query Params
        category_index = req.get_param_as_int('categoryIndex')
        threshold = req.get_param('threshold')
        get_multilabel_results = req.get_param_as_bool('getMultilabelResults')
        get_combined_results = req.get_param_as_bool('getCombinedResults')
        get_layer_output = req.get_param_as_bool('getLayerOutput')
        get_all_graylevels = req.get_param_as_bool('getAllGrayLevels')
        get_category_graylevel = req.get_param_as_bool('getCategoryGraylevel')

        image_url = req.get_param('imageUrl')

        try:
            if image_url:
                img = image_url # Utils.get_cv2_img_array(image_url)
            else:
                data = msgpack.loads(req.stream.read())
                img = data.get("image")


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
                combined_output = None
                combined_output = neurodoll.combine_neurodoll_v3labels_and_multilabel(img)
                ret['combined_output'] = combined_output
                ret['mask'] = combined_output
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
        return(ret)

api = falcon.API()
api.add_route('/nd/', NeurodollResource())
