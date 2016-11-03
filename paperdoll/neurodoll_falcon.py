__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
import traceback
import falcon
from .. import neurodoll, neurodoll_single_category
from .. import neurodoll_with_multilabel
from .. import constants
# from .darknet.pyDarknet import mydet

from jaweson import json, msgpack

print "Done with imports"

class PaperResource:
    def __init__(self):
        print "Loaded Resource"


    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        print "Reached on_post"
        category_index = req.get_param('categoryIndex')
        category_index = category_index and int(category_index)

        get_multilabel_results = req.get_param('getMultilabelResults')
        print('get multi:'+str(get_multilabel_results))
        get_multilabel_results = get_multilabel_results == "true" or get_multilabel_results == "True" or get_multilabel_results == True
        # get_multilabel_results = True if get_multilabel_results in ["true", "True", True] else False

        get_combined_results = req.get_param('getCombinedResults')
        print('get combined:'+str(get_combined_results))
        get_combined_results = get_combined_results == "true" or get_combined_results == "True" or get_combined_results == True

        get_layer_output = req.get_param('getLayerOutput')
        print('get layer output:'+str(get_layer_output))
#        if get_layer_output == "true" or get_layer_output == "True" or get_layer_output == True:
#            get_layer_output = 'myfc7'

        get_all_graylevels = req.get_param('getAllGrayLevels')
        print('get all graylevels:'+str(get_all_graylevels))
        if get_all_graylevels == "true" or get_all_graylevels == "True" or get_all_graylevels == True:
            get_all_graylevels = True

        get_category_graylevel = req.get_param('getCategoryGraylevel')
        print('get graylevel:'+str(get_category_graylevel))
        #get_multilabel_results = get_multilabel_results == "true" or get_multilabel_results == "True" or get_multilabel_results == True
        # get_multilabel_results = True if get_multilabel_results in ["true", "True", True] else False

#        get_yolo_results = req.get_param('getYolo')
#        print('get yolo:'+str(get_yolo_results))
#        get_yolo_results = get_yolo_results == "true" or get_yolo_results == "True" or get_yolo_results == True

        ret = {"success": False}

        try:
            data = msgpack.loads(req.stream.read())
            img = data.get("image")

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
                combined_output = neurodoll.combine_neurodoll_and_multilabel(img)
                ret['combined_output'] = combined_output
                if combined_output is not None:
                    ret["success"] = True

        # yonti style - single category mask
            ret["label_dict"] = constants.ultimate_21_dict
            if category_index:
                ret["mask"] = neurodoll.get_category_graylevel(img, category_index)
                if ret["mask"] is not None:
                    ret["success"] = True

        # layer output for yonti - default is last fc layer (myfc7) but any can be accessed (put layer name as argument)
            if get_layer_output:
                ret["layer_output"] = neurodoll.get_layer_output(get_layer_output)
                if ret["layer_output"] is not None:
                    ret["success"] = True
                else:
                    ret["error"] = "no layer output obtained"

        # regular neurodoll call
            if not get_multilabel_results and not get_combined_results and not category_index:
                print "No special params, inferring..."
                ret["mask"] = neurodoll.infer_one(img)
                if ret["mask"] is not None:
                    ret["success"] = True
                else:
                    ret["error"] = "No mask from ND"


        except Exception as e:
            traceback.print_exc()
            ret["error"] = traceback.format_exc()

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/nd/', PaperResource())
