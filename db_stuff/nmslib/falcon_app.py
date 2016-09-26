import falcon
import argparse
from jaweson import json, msgpack
from os import environ
import load_n_search
import re

class Search:
    def __init__(self, collection_name, category_name, index_ver):
        self.collection = collection_name
        self.category = category_name
        self.index_version = index_ver
        index, nmslib_vector = load_n_search.load_index(collection_name, category_name, index_ver)
        self.index = index
        self.nmslib_vector = nmslib_vector


    def on_get(self, resp):
        print("got GET")
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        print('got POST')
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            fp = data.get("fp")
            k = data.get("k")
            ret["data"] = load_n_search.find_to_k(fp, k, self.nmslib_vector, self.index)
            ret["success"] = True
        except Exception as e:
            ret["error"] = str(e)
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


# def get_user_input():
#     parser = argparse.ArgumentParser(description='"@@@ nmslib falcon @@@')
#     parser.add_argument('-n', '--name', required=True, dest="col_name",
#                         help='collection name - without gender or countycode')
#     parser.add_argument('-c', '--code', defualt='US', dest="country_code",
#                         help='country code - currently doing only US or DE')
#     parser.add_argument('-g', '--gender', dest="gender", choices=['Female', 'Male'],
#                         help='specify which gender to index (Female or Male)')
#     parser.add_argument('-p', '--part', dest="category", required=True,
#                         help='which category to index')
#     args = parser.parse_args()
#     return args
#
#
# user_input = get_user_input()
# col_name = user_input.col_name
# cc = user_input.country_code
# gender = user_input.gender
# category = user_input.category
#
# collection = '%s_%s_%s' % (col_name, cc, gender)
# route = '/'+category
#
# api = falcon.API()
# api.add_route(route, Search(collection, category))


# import argparse
# import json
#
# import falcon
#
# from gunicorn.app.base import BaseApplication
# from gunicorn.six import iteritems

#
# class StandaloneApplication(BaseApplication):
#
#     def __init__(self, app, options=None):
#         self.options = options or {}
#         self.application = app
#         super(StandaloneApplication, self).__init__()
#
#     def load_config(self):
#         config = dict(
#             [(key, value) for key, value in iteritems(self.options)
#                 if key in self.cfg.settings and value is not None]
#         )
#         for key, value in iteritems(config):
#             self.cfg.set(key.lower(), value)
#
#     def load(self):
#         return self.application


# def _get_config():
#     parser = argparse.ArgumentParser(description='"@@@ nmslib falcon @@@')
#     parser.add_argument('-n', '--name', required=True, dest="col_name",
#                         help='collection name - without gender or countycode')
#     parser.add_argument('-c', '--code', default='US', dest="country_code",
#                         help='country code - currently doing only US or DE')
#     parser.add_argument('-g', '--gender', dest="gender", choices=['Female', 'Male'],
#                         help='specify which gender to index (Female or Male)')
#     parser.add_argument('-s', '--select', dest="category", required=True,
#                         help='which category to index')
#     args = parser.parse_args()
#
#     return args
inputs = environ.get('NMSLIB_INPUTS')
user_inputs = re.split(r'/', inputs)

collection = user_inputs[0]
categories = user_inputs[1]
index_version = user_inputs[2]
api = falcon.API()
#     # api.add_route('/', ExampleResource())
#
#     # user_input, config = _get_config()
#     # col_name = user_input.col_name
#     # cc = user_input.country_code
#     # gender = user_input.gender
#     # category = user_input.category
#     #
# collection = '%s_%s_%s' % (col_name, cc, gender)
# for category in categories:
#     print(category)
route = '/'+categories
api.add_route(route, Search(collection, categories,index_version))



