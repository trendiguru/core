__author__ = 'liorsabag'
# labels for pixel level parsing (neurodoll) are in constants.ultimate21 (21 labels)
# labels for multilabel image-level categorization are in constants.web_tool_categories (also 21 labels)
import falcon
from .. import neurodoll, neurodoll_single_category
from .. import multilabel
from .. import constants

from jaweson import json, msgpack


class PaperResource:
    def on_get(self, req, resp):
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        category_index = req.get_param('categoryIndex')
        category_index = category_index and int(category_index)

        get_multilabel_results = req.get_param('getMultilabelResults')
        print('get multi:'+str(get_multilabel_results))
        get_multilabel_results = get_multilabel_results == "true" or get_multilabel_results == "True" or get_multilabel_results == True


        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            img = data.get("image")

            if get_multilabel_results:
                output = multilabel.get_multilabel_output(img)
                ret['multilabel_output'] = output
                print('multilabel output:'+str(output))
            if category_index:
                ret["mask"] = neurodoll_single_category.get_category_graylevel(img, category_index) 
            else:
                ret["mask"] = neurodoll.infer_one(img)
            
            ret["label_dict"] = constants.ultimate_21_dict
            if ret["mask"] is not None:
                ret["success"] = True
            else:
                ret["error"] = "No mask from ND"

        except Exception as e:
            ret["error"] = str(e)

        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/nd/', PaperResource())
