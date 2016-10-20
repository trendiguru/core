import falcon
from jaweson import json, msgpack
from .. import edit_results
from bson import json_util

# Logging
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
USER_FILTER = 'stylebook'


class Editor(object):

    def on_get(self, req, resp, **path_args):
        logging.debug("ON_GET {0}".format(str(path_args)))
        ret = {'ok': False, 'data': {}}
        params = req.params
        try:
            if 'image_id' in path_args:
                ret['data'] = edit_results.get_image_obj_for_editor(None, image_id=path_args["image_id"])
                ret['ok'] = True
            if 'image_url' in params:
                url = params["image_url"]
                ret['data'] = edit_results.get_image_obj_for_editor(url)
                ret['ok'] = True
            elif 'last' in params:
                amount = int(params['last'])
                # user_filter = req.USER
                ret['data'] = edit_results.get_latest_images(amount, user_filter=USER_FILTER)
                ret['ok'] = True
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)

    def on_delete(self, req, resp, **path_args):
        logging.debug("ON_DELETE {0}".format(str(path_args)))
        ret = {'ok': False, 'data': {}}
        try:
            if "result_id" in path_args:
                ret['ok'] = edit_results.cancel_result(path_args["image_id"],
                                                       path_args["person_id"],
                                                       path_args["item_category"],
                                                       path_args["results_collection"],
                                                       path_args["result_id"])
            elif "item_category" in path_args:
                ret['ok'] = edit_results.cancel_item(path_args["image_id"],
                                                     path_args["person_id"],
                                                     path_args["item_category"])
            elif "person_id" in path_args:
                ret['ok'] = edit_results.cancel_person(path_args["image_id"],
                                                       path_args["person_id"])
            else:
                ret['ok'] = edit_results.cancel_image(path_args["image_id"])
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)

    def on_put(self, req, resp, **path_args):
        ret = {'ok': False, 'data': {}}
        body = req.stream.read()
        if body:
            data = json_util.loads(body)['data']
        try:
            if "results_collection" in path_args:
                ret['ok'] = edit_results.reorder_results(path_args["image_id"],
                                                         path_args["person_id"],
                                                         path_args["item_category"],
                                                         path_args["results_collection"],
                                                         data)
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)

    def on_patch(self, req, resp, **path_args):
        ret = {'ok': False, 'data': {}}
        body = req.stream.read()
        if body:
            data = json_util.loads(body)['data']
        try:
            if "person_id" in path_args:
                gender = data['gender'] if 'gender' in data.keys() else data
                ret["ok"] = edit_results.change_gender_and_rebuild_person(path_args["image_id"],
                                                                          path_args["person_id"],
                                                                          gender)
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)


