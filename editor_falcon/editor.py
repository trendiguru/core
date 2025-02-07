import traceback
import falcon
from jaweson import json, msgpack
from .. import edit_results, page_results, constants
from bson import json_util
from . import temp_editor_users

# Logging
import logging
import sys

# root = logging.getLogger()
# root.setLevel(logging.DEBUG)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# root.addHandler(ch)

USER_FILTER = 'stylebook'
db = constants.db


class Editor(object):

    def on_get(self, req, resp, **path_args):
        logging.debug("ON_GET {0}".format(str(path_args)))
        
        user_identifier = req.context["user_identifier"]
        logging.debug("user_identifier: {0}".format(user_identifier))
        
        ret = {'ok': False, 'data': {}}
        params = req.params
        try:
            if 'last' in params:
                amount = int(params['last'])
                # user_filter = req.USER
                ret['data'] = edit_results.get_latest_images(amount, user_filter=USER_FILTER)
                ret['ok'] = True
            else:
                # Exactly one of these should have a non-None value.
                url = params.get("image_url")
                image_id = path_args.get("image_id")
                if bool(url) == bool(image_id):
                    raise ValueError("Please provide either image_url or image_id (and not both)")

                collections = []
                if not params.get("multi_collection") in [True, "True", "true"]:
                    collections.append(temp_editor_users.USERS[user_identifier]["default_collection"])
                ret['data'] = edit_results.get_image_obj_for_editor(url, image_id=image_id, product_collections=collections)
                ret['ok'] = True

            assert ret['ok']
            resp.status = falcon.HTTP_200
        
        except Exception as e:
            ret['error'] = traceback.format_exc()
            resp.status = falcon.HTTP_400
        
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
            
            assert ret['ok']
            resp.status = falcon.HTTP_200
            
        except Exception as e:
            ret['error'] = traceback.format_exc()
            resp.status = falcon.HTTP_400
        
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
            assert ret['ok']
            resp.status = falcon.HTTP_202 # Accepted
        
        except Exception as e:
            ret['error'] = traceback.format_exc()
            resp.status = falcon.HTTP_400
        
        resp.body = json_util.dumps(ret)

    def on_patch(self, req, resp, **path_args):
        ret = {'ok': False, 'data': {}}
        body = req.stream.read()
        if body:
            data = json_util.loads(body)
        try:
            if "person_id" in path_args:
                gender = data['gender'] if 'gender' in data.keys() else data['data']
                ret["ok"] = edit_results.change_gender_and_rebuild_person(path_args["image_id"],
                                                                          path_args["person_id"],
                                                                              gender)
            assert ret['ok']
            resp.status = falcon.HTTP_202 # Accepted
        except Exception as e:
            ret['error'] = traceback.format_exc()
            resp.status = falcon.HTTP_400
        
        resp.body = json_util.dumps(ret)

    def on_post(self, req, resp, **path_args):
        ret = {'ok': False, 'data': {}}
        body = req.stream.read()
        if body:
            data = json_util.loads(body)
        user_email = req.context["user_identifier"]
        pid = db.users.find_one({'email': user_email})['pid']
        products_collection = page_results.get_collection_from_ip_and_pid(None, pid)
        try:
            if "results_collection" in path_args:
                ret['ok'] = edit_results.add_result(path_args['image_id'],
                                                    path_args["person_id"],
                                                    path_args["item_category"],
                                                    path_args["results_collection"],
                                                    data)
            elif "person_id" in path_args:
                data = data.get("data", data)
                ret['ok'] = edit_results.add_item(path_args["image_id"],
                                                  path_args["person_id"],
                                                  data["category"],
                                                  products_collection)
            elif "image_id" in path_args:
                ret['ok'] = edit_results.add_people_to_image(path_args['image_id'],
                                                             data['faces'],
                                                             products_collection,
                                                             'pd')
            assert ret['ok']
            resp.status = falcon.HTTP_201 # Created
        
        except Exception as e:
            ret['error'] = traceback.format_exc()
            resp.status = falcon.HTTP_400
        
        resp.body = json_util.dumps(ret)


