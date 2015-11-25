__author__ = 'yonatan'

"""
the download worker
"""

import json
import collections
import urllib
import time

import requests
from rq import Queue

from fingerprint_core import generate_mask_and_insert
import constants

q = Queue('fingerprint_new', connection=constants.redis_conn)

BASE_URL = "http://api.shopstyle.com/api/v2/"
BASE_URL_PRODUCTS = BASE_URL + "products/"
PID = "uid900-25284470-95"
TG_FIELDS = ["fp_version", "fingerprint", "fp_date", "human_bb", "bounding_box"]
# ideally, get filter list from DB. for now:
# removed "Category" for now because its results  are not disjoint (sum of subsets > set)
FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
MAX_RESULTS_PER_PAGE = 50
MAX_OFFSET = 5000
MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE
db = constants.db
# collection = constants.update_collection_name
relevant = constants.db_relevant_items
fp_version = constants.fingerprint_version


def download_products(filter_params, total=MAX_SET_SIZE, coll="products"):
    # if not isinstance(filter_params, UrlParams):
    #     filter_params = UrlParams(params_dict=filter_params)
    #
    # dl_query = {"filter_params": filter_params.encoded()}
    #
    # if db.dl_cache.find_one(dl_query):
    #     print "We've done this batch already, let's not repeat work"
    #     return
    collection = coll
    if collection != "products":
        filter_params["site"] = "www.shopstyle.co.jp"
    if "filters" in filter_params:
        del filter_params["filters"]
    filter_params["limit"] = MAX_RESULTS_PER_PAGE
    filter_params["offset"] = 0
    while filter_params["offset"] < MAX_OFFSET and \
                    (filter_params["offset"] + MAX_RESULTS_PER_PAGE) <= total:
        product_response = delayed_requests_get(BASE_URL_PRODUCTS, filter_params, collection)
        product_results = product_response.json()
        total = product_results["metadata"]["total"]
        products = product_results["products"]
        for prod in products:
            db_update(prod, collection)
        filter_params["offset"] += MAX_RESULTS_PER_PAGE

    # Write down that we did this
    # db.dl_cache.insert(dl_query)

    print "Batch Done. Total Product count: {0}".format(db[collection].count())


def insert_and_fingerprint(prod, collection, current_date):
    """
    this func. inserts a new product to our DB and runs TG fingerprint on it
    :param prod: dictionary of shopstyle product
    :return: Nothing, void function
    """

    print "enqueuing for fingerprint & insert,",
    q.enqueue(generate_mask_and_insert, doc=prod, image_url=None,
              fp_date=current_date, coll=collection)


def db_update(prod, collection):
    print " Updating product {0}. ".format(prod["id"]),
    current_date = db.download_data.find({"criteria": collection})[0]["current_dl"]

    # requests package can't handle https - temp fix
    prod["image"] = json.loads(json.dumps(prod["image"]).replace("https://", "http://"))
    prod_in_que = db.fp_in_process.find_one({"id": prod["id"]})
    if prod_in_que is not None:
        return
    db.download_data.find_one_and_update({"criteria": collection},
                                         {'$inc': {"items_downloaded": 1}})
    prod["download_data"] = {"dl_version": current_date}
    # case 1: new product - try to update, if does not exists, insert a new product and add our fields
    prod_in_coll = db[collection].find_one({"id": prod["id"]})
    category = prod['categories'][0]['id']
    print category
    if prod_in_coll is None:
        print "Product not in db." + collection
        # case 1.1: try finding this product in the products
        if collection != "products":
            prod_in_prod = db.products.find_one({"id": prod["id"]})
            if prod_in_prod is not None:
                print "but new product is already in db.products"
                prod["fingerprint"] = prod_in_prod["fingerprint"]
                prod["download_data"] = prod_in_prod["download_data"]
                db[collection].insert_one(prod)
                print "prod inserted successfully to " + collection
                return
        # case 1.2: try finding this product in the products
        print ", searching in archive. "
        prod_in_archive = db.archive.find_one({'id': prod["id"]})
        if prod_in_archive is None:
            print "New product,",
            db.download_data.find_one_and_update({"criteria": collection},
                                                 {'$inc': {"new_items": 1}})
            db.fp_in_process.insert_one({"id": prod["id"]})
            insert_and_fingerprint(prod, collection, current_date)
        else:  # means the item is in the archive
            # No matter what, we're moving this out of the archive...
            prod_in_archive["archive"] = False
            print "Prod in archive, checking fingerprint version...",
            if prod_in_archive.get("download_data")["fp_version"] == constants.fingerprint_version:
                print "fp_version good, moving from db.archive to db.products",
                prod_in_archive.update(prod)
                db[collection].insert_one(prod_in_archive)
            else:
                print "old fp_version, updating fp",
                insert_and_fingerprint(prod, collection, current_date)
                # db.archive.delete_one({'id': prod["id"]})
                # db.download_data.find_one_and_update({"criteria": collection},
                #                                      {'$inc': {"returned_from_archive": 1}})
    else:
        # case 2: the product was found in our db, and maybe should be modified
        print "Found existing prod in db,",
        if prod_in_coll.get("download_data")["fp_version"] == fp_version:
            # Thus - update only shopstyle's fields
            # db.download_data.find_one_and_update({"criteria": collection},
            #                                      {'$inc': {"existing_items": 1}})
            db[collection].update_one({'id': prod["id"]},
                                   {'$set': {'download_data.dl_version': current_date}})
        else:
            # db.download_data.find_one_and_update({"criteria": collection},
            #                                      {'$inc': {"existing_but_renewed": 1}})
            db[collection].delete_one({'id': prod['id']})
            insert_and_fingerprint(prod, collection, current_date)
            print "product with an old fp was refingerprinted"


def delayed_requests_get(url, _params, collection):
    dl_data = db.download_data.find({"criteria": collection})[0]
    sleep_time = max(0, 0.1 - (time.time() - dl_data["last_request"]))
    time.sleep(sleep_time)
    db.download_data.find_one_and_update({"criteria": collection}, {'$set': {"last_request": time.time()}})
    return requests.get(url, params=_params)


class UrlParams(collections.MutableMapping):
    """This is current designed specifically for the ShopStyle API where there can be multiple "fl" parameters,
    although it could easily be made more modular, where a list of keys which can have multiple values is predefined,
    or the option (bool multivalue) is supplied on new entry creation...
    """

    def __init__(self, params_dict={}, _fl_list=[]):
        self.p_dict = dict(params_dict)
        self.fl_list = list(_fl_list)

    def __iter__(self):
        for key in self.p_dict:
            yield key
        for i in range(0, len(self.fl_list)):
            yield ("fl", i)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.fl_list[key[1]]
        elif key == "fl":
            return self.fl_list
        else:
            return self.p_dict[key]

    def __setitem__(self, key, value):
        if key == "fl":
            self.fl_list.append(value)
        else:
            self.p_dict[key] = value

    def __delitem__(self, key):
        del self.p_dict[key]

    def __len__(self):
        return len(self.p_dict) + len(self.fl_list)

    def iteritems(self):
        for k, v in self.p_dict.iteritems():
            yield k, v
        for item in self.fl_list:
            yield 'fl', item

    def items(self):
        return [(k, v) for k, v in self.iteritems()]

    def copy(self):
        return UrlParams(self.p_dict, self.fl_list)

    @staticmethod
    def encode_params(data):
        """Encode parameters in a piece of data.
        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.

        Taken from requests package:
        https://github.com/kennethreitz/requests/blob/8b5e457b756b2ab4c02473f7a42c2e0201ecc7e9/requests/models.py
        """

        def to_key_val_list(value):
            """Take an object and test to see if it can be represented as a
            dictionary. If it can be, return a list of tuples, e.g.,
            ::
                >>> to_key_val_list([('key', 'val')])
                [('key', 'val')]
                >>> to_key_val_list({'key': 'val'})
                [('key', 'val')]
                >>> to_key_val_list('string')
                ValueError: cannot encode objects that are not 2-tuples.
            """
            if value is None:
                return None

            if isinstance(value, (str, bytes, bool, int)):
                raise ValueError('cannot encode objects that are not 2-tuples')

            if isinstance(value, collections.Mapping):
                value = value.items()

            return list(value)

        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, 'read'):
            return data
        elif hasattr(data, '__iter__'):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, '__iter__'):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (k.encode('utf-8') if isinstance(k, str) else k,
                             v.encode('utf-8') if isinstance(v, str) else v))
            return urllib.urlencode(result, doseq=True)
        else:
            return data

    def encoded(self):
        return self.__class__.encode_params(self)
