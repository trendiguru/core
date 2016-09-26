'''
1. finds open port
2. opens thread with gunicorn listening on that port
3. gets back process id
4. keeps track of all gunis and ports and pids
5. when needs to refresh -
    opens new guni
    builds index with nmslib_index_1/2
    when ready changes lookup table and closes old port
'''

import socket
import subprocess
import build_index
import falcon
from jaweson import msgpack
import requests
import psutil
import json

SERVER = "http://extremeli-evolution-dev-2:"  # use the name of the server running the gunicorn
lookup_table = {}


def nmslib_find_top_k(fp, k, port, category):
    data = msgpack.dumps({"fp": fp,
                          "k": k})
    category_server = SERVER+str(port)+'/'+category
    resp = requests.post(category_server, data=data)
    return msgpack.loads(resp.content)


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def fill_lookup_table(collections):
    for collection in collections:
        lookup_table[collection]={}
        if 'recruit' in collection:
            from trendi.db_stuff.recruit.recruit_constants import recruit2category_idx
            categories_female = categories_male = list(set(recruit2category_idx.keys()))
        elif 'amaz' in collection:
            from trendi.db_stuff.amazon.amazon_constants import amazon_categories_list
            categories_female = categories_male = amazon_categories_list
        else:
            from trendi.db_stuff.shopstyle.shopstyle_constants import shopstyle_paperdoll_female, shopstyle_paperdoll_male
            categories_female = list(set(shopstyle_paperdoll_female.values()))
            categories_male = list(set(shopstyle_paperdoll_male.values()))
        categories_male.sort()
        if 'Female' in collection:
            cats = categories_female
        elif 'Male' in collection:
            cats = categories_male
        else:
            Exception
        cats.sort()
        for category in cats:
            p = find_free_port()
            process = 'gunicorn -b :%s falcon_app:api --env NMSLIB_INPUTS=%s/%s/1 -w 2 --preload' \
                      % (p, collection, category)
            pid = subprocess.Popen(process, shell=True)
            lookup_table[collection][category] = {'port': p,
                                                  'pid': pid,
                                                  'index_version': 1}


def rebuild_index(collection_name, category):
    global lookup_table
    current_index_version = lookup_table[collection_name][category]['index_version']
    if current_index_version == 1:
        new_version = 2
    else:
        new_version = 1
    p = find_free_port()
    build_index.build_n_save(collection_name,category)
    process = 'gunicorn -b :%s falcon_app:api --env NMSLIB_INPUTS=%s/%s/%d -w 2 --preload' \
              % (p, collection_name, category, new_version)
    pid_new = subprocess.Popen(process)
    pid_old = lookup_table[collection_name][category]['pid']
    lookup_table[collection_name][category] = {'port': p,
                                               'pid': pid_new,
                                               'index_version': new_version}

    process_2_terminate = psutil.Process(pid_old)
    process_2_terminate.terminate()


class Selector:
    def on_get(self, resp):
        print("got GET")
        """Handles GET requests"""
        quote = {
            'quote': 'I\'ve always been more interested in the future than in the past.',
            'author': 'Grace Hopper'
        }

        resp.body = json.dumps(quote)

    def on_post(self, req, resp):
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            collection = data.get("collection")
            category = data.get("category")
            fp = data.get("fp")
            port = lookup_table[collection][category]['port']
            print(port)
            ret = nmslib_find_top_k(fp, 1000, port, category)
            print ('done')
        except Exception as e:
            ret["error"] = str(e)
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200

    def on_put(self, req, resp):
        ret = {"success": False}
        try:
            data = msgpack.loads(req.stream.read())
            collection_name = data.get("collection")
            category = data.get("category")
            rebuild_index(collection_name, category)
            ret["success"] = True
        except Exception as e:
            ret["error"] = str(e)
        resp.data = msgpack.dumps(ret)
        resp.content_type = 'application/x-msgpack'
        resp.status = falcon.HTTP_200


fill_lookup_table(['recruit_Male'])
api = falcon.API()
api.add_route('/bouncer', Selector())

