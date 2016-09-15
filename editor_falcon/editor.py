import falcon
from jaweson import json, msgpack
from .. import edit_results
from bson import json_util


class Editor(object):

    def on_get(self, req, resp, **kwargs):
        ret = {'ok': False, 'data': {}}
        params = req.params
        print params
        try:
            if "image_id" in kwargs:
                print "Getting your image for you..."
                ret['data'] = edit_results.get_image_obj_for_editor(None, id=kwargs["image_id"])
                ret['ok'] = True
            if 'image_url' in params:
                url = params["image_url"]
                ret['data'] = edit_results.get_image_obj_for_editor(url)
                ret['ok'] = True
            elif 'last' in params:
                print "last = {0}".format(str(req.env))
                amount = params['last']
                ret['data'] = edit_results.get_latest_images(amount)
                ret['ok'] = True
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)

    def on_delete(self, req, resp, **kwargs):
        ret = {'ok': False, 'data': {}}
        try:
            if "result_id" in kwargs:
                ret['ok'] = edit_results.cancel_result(kwargs["image_id"],
                                                       kwargs["person_id"],
                                                       kwargs["item_category"],
                                                       kwargs["results_collection"],
                                                       kwargs["result_id"])
            elif "item_category" in kwargs:
                ret['ok'] = edit_results.cancel_item(kwargs["image_id"],
                                                     kwargs["person_id"],
                                                     kwargs["item_category"])
            elif "person_id" in kwargs:
                ret['ok'] = edit_results.cancel_person(kwargs["image_id"],
                                                       kwargs["person_id"])
            else:
                ret['ok'] = edit_results.cancel_image(kwargs["image_id"])
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)

    def on_put(self, req, resp, **kwargs):
        ret = {'ok': False, 'data': {}}
        body = req.stream.read()
        if body:
            data = json_util.loads(body)['data']
        try:
            if "results_collection" in kwargs:
                ret['ok'] = edit_results.reorder_results(kwargs["image_id"],
                                                         kwargs["person_id"],
                                                         kwargs["item_category"],
                                                         kwargs["results_collection"],
                                                         body["data"])
            elif "person_id" in kwargs:
                ret["ok"] = edit_results.change_gender_and_rebuild_person(kwargs["image_id"],
                                                                          kwargs["person_id"],
                                                                          data["gender"])
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)

    def on_patch(self, req, resp, **kwargs):
        ret = {'ok': False, 'data': {}}
        body = req.stream.read()
        if body:
            data = json_util.loads(body)['data']
        try:
            if "person_id" in kwargs:
                ret["ok"] = edit_results.change_gender_and_rebuild_person(kwargs["image_id"],
                                                                          kwargs["person_id"],
                                                                          data["gender"])
        except Exception as e:
            ret['error'] = str(e)
        resp.status = falcon.HTTP_200
        resp.body = json_util.dumps(ret)


