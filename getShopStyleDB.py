import collections
import time
import json
import urllib

import requests
from redis import Redis
from rq import Queue

from fingerprint_core import generate_mask_and_insert
import constants
import find_similar_mongo

redis_conn = Redis()

q = Queue('fingerprint', connection=redis_conn)

BASE_URL = "http://api.shopstyle.com/api/v2/"
BASE_URL_PRODUCTS = BASE_URL + "products/"
PID = "uid900-25284470-95"
TG_FIELDS = ["fp_version", "fingerprint", "fp_date", "human_bb", "bounding_box"]
# ideally, get filter list from DB. for now:
# removed "Category" for now because its results  are not disjoint (sum of subsets > set)
FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
MAX_RESULTS_PER_PAGE = 50
MAX_OFFSET = 1000
MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE
db = constants.db_name
collection = constants.update_collection


class ShopStyleDownloader():
    def __init__(self):
        # connect to db
        self.db = db
        self.collection = self.db[collection]
        self.current_dl_version = constants.download_version
        self.last_request_time = time.time()
        self.do_fingerprint = True

    def _run_by_filter(self):
        # Let's try an initial category ("womens-suits")
        initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": "womens-suits", "Sort": "Recency"})
        self.divide_and_conquer(initial_filter_params, 0)

    def run_by_category(self, root_id_list=None, do_fingerprint=True):
        self.do_fingerprint = do_fingerprint
        root_category, ancestors = self.build_category_tree()

        if isinstance(root_id_list, basestring):
            root_id_list = [root_id_list]
        # if root_id_list not given, special case when we want to DL everything, because the root_category doesn't exit.
        cats_to_dl = root_id_list or [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            self.download_category(cat)

        db.dl_cache.remove()
        print "DONE!!!!!"

    def build_category_tree(self):
        # download all categories
        category_list_response = requests.get(BASE_URL + "categories", params={"pid": PID})
        category_list_response_json = category_list_response.json()
        root_category = category_list_response_json["metadata"]["root"]["id"]
        category_list = category_list_response_json["categories"]
        self.db.categories.remove({})
        self.db.categories.insert(category_list)
        # find all the children
        for cat in self.db.categories.find():
            self.db.categories.update({"id": cat["parentId"]}, {"$addToSet": {"childrenIds": cat["id"]}})
        # get list of all categories under root - "ancestors"
        ancestors = []
        for c in self.db.categories.find({"parentId": root_category}):
            ancestors.append(c)
        # let's get some numbers in there - get a histogram for each ancestor
        for anc in ancestors:
            response = self.delayed_requests_get(BASE_URL_PRODUCTS + "histogram",
                                                 {"pid": PID, "filters": "Category", "cat": anc["id"]})
            hist = response.json()["categoryHistogram"]
            # save count for each category
            for cat in hist:
                self.db.categories.update({"id": cat["id"]}, {"$set": {"count": cat["count"]}})
        return root_category, ancestors

    def download_category(self, category_id):
        category = self.db.categories.find_one({"id": category_id})
        if "count" in category and category["count"] <= MAX_SET_SIZE:
            print("Attempting to download: {0} products".format(category["count"]))
            print("Category: " + category_id)
            self.download_products({"pid": PID, "cat": category_id})
        elif "childrenIds" in category.keys():
            print("Splitting {0} products".format(category["count"]))
            print("Category: " + category_id)
            for child_id in category["childrenIds"]:
                self.download_category(child_id)
        else:
            initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": category["id"]})
            self.divide_and_conquer(initial_filter_params, 0)
        self.archive_products(category_id)

    def divide_and_conquer(self, filter_params, filter_index):
        """Keep branching until we find disjoint subsets which have less then MAX_SET_SIZE items"""
        if filter_index >= len(FILTERS):
            # TODO: determine appropriate behavior in this case
            print "ran out of FILTERS"
            return
        filter_params["filters"] = FILTERS[filter_index]

        histogram_response = self.delayed_requests_get(BASE_URL_PRODUCTS + "histogram", filter_params)
        # TODO: check the request worked here
        try:
            histogram_results = histogram_response.json()
            # for some reason Category doesn't come with a prefix...

            filter_prefix = histogram_results["metadata"].get("histograms")[0].get("prefix")

            hist_key = FILTERS[filter_index].lower() + "Histogram"
            if hist_key not in histogram_results:
                raise RuntimeError("No hist_key {0} in histogram: \n{1}".format(hist_key, histogram_results))
        except:
            print "Could not get histogram for filter_params: {0}".format(filter_params.encoded())
            print "Attempting to downloading first 5000"
            # TODO: Better solution for this
            self.download_products(filter_params)
        else:
            hist_key = FILTERS[filter_index].lower() + "Histogram"
            for subset in histogram_results[hist_key]:
                subset_filter_params = filter_params.copy()
                if FILTERS[filter_index] == "Category":
                    _key = "cat"
                    filter_prefix = ""
                else:
                    _key = "fl"
                subset_filter_params[_key] = filter_prefix + subset["id"]
                if subset["count"] < MAX_SET_SIZE:
                    # print("Attempting to download: %i products" % subset["count"])
                    # print("Params: " + str(subset_filter_params.items()))
                    # pdb.set_trace()
                    self.download_products(subset_filter_params)
                else:
                    print "Splitting: {0} products".format(subset["count"])
                    print "Params: {0}".format(subset_filter_params.encoded())
                    self.divide_and_conquer(subset_filter_params, filter_index + 1)

    def download_products(self, filter_params, total=MAX_SET_SIZE):
        """
        Download with paging...
        :param filter_params:
        :param total:
        """

        if not isinstance(filter_params, UrlParams):
            filter_params = UrlParams(params_dict=filter_params)

        dl_query = {"dl_version": self.current_dl_version,
                    "filter_params": filter_params.encoded()}

        if self.db.dl_cache.find_one(dl_query):
            print "We've done this batch already, let's not repeat work"
            return

        if "filters" in filter_params:
            del filter_params["filters"]
        filter_params["limit"] = MAX_RESULTS_PER_PAGE
        filter_params["offset"] = 0
        while filter_params["offset"] < MAX_OFFSET and \
                        (filter_params["offset"] + MAX_RESULTS_PER_PAGE) <= total:
            product_response = self.delayed_requests_get(BASE_URL_PRODUCTS, filter_params)
            product_results = product_response.json()
            total = product_results["metadata"]["total"]
            products = product_results["products"]
            for prod in products:
                self.db_update(prod)
            filter_params["offset"] += MAX_RESULTS_PER_PAGE

        # Write down that we did this
        self.db.dl_cache.insert(dl_query)

        print "Batch Done. Total Product count: {0}".format(self.collection.count())

    def archive_products(self, category_id=None):
        print "archiving old products"
        # this process iterate our whole DB (after daily download) and shifts unwanted items to the archive collection
        query_doc = {'$and': [
            {'dl_version': {'$exists': 1, '$lt': self.current_dl_version}},
            {'$or': [
                {'archive': {'$exists': 0}},
                {'archive': False}
            ]}]}

        if category_id:
            query_doc["$and"].append(
                {"categories":
                     {"$elemMatch": {
                         "id": {"$in": find_similar_mongo.get_all_subcategories(self.db.categories, category_id)}}}
                 })

        update_result = self.collection.update_many(query_doc, {"$set": {"archive": True}})
        # for prod in old_prods:
        #     self.db.archive.insert(prod)
        #     self.db.products.delete_one({"id": prod["id"]})
        print "Marked {0} of {1} products for archival".format(update_result.modified_count,
                                                               update_result.matched_count)

    def delayed_requests_get(self, url, _params):
        sleep_time = max(0, 0.1 - (time.time() - self.last_request_time))
        time.sleep(sleep_time)
        self.last_request_time = time.time()
        return requests.get(url, params=_params)

    def shopstyle_fields_update(self, prod):
        """
        this func. updates only shopstyle's fields of a product in the DB
        :param prod: dictionary of a DB product
        :return: Nothing, void function
        """
        self.collection.update_one({'id': prod['id']}, {'$set': prod})

    def insert_and_fingerprint(self, prod, do_fingerprint=None):
        """
        this func. inserts a new product to our DB and runs TG fingerprint on it
        :param prod: dictionary of shopstyle product
        :return: Nothing, void function
        """
        if do_fingerprint is None:
            do_fingerprint = self.do_fingerprint

        prod["_id"] = self.collection.insert_one(prod).inserted_id
        if do_fingerprint:
            print "enqueuing for fingerprinting...,",
            q.enqueue(generate_mask_and_insert, image_url=None, doc=prod, save_to_db=True, mask_only=False)
            # prod_fp = super_fp(image_url=None, db_doc=prod, )
            # prod["fingerprint"] = prod_fp
            # prod["fp_version"] = constants.fingerprint_version
        print "inserting,",

    def db_update(self, prod):
        print ""
        print "Updating product {0}. ".format(prod["id"]),

        # requests package can't handle https - temp fix
        prod["image"] = json.loads(json.dumps(prod["image"]).replace("https://", "http://"))
        prod["dl_version"] = self.current_dl_version

        # case 1: new product - try to update, if does not exists, insert a new product and add our fields
        prod_in_products = self.collection.find_one({"id": prod["id"]})

        if prod_in_products is None:
            print "Product not in db.products, searching in archive. ",
            self.insert_and_fingerprint(prod)
            # case 1.1: try finding this product in the archive
            # prod_in_archive = noneself.db.archive.find_one({'id': prod["id"]})
            # if prod_in_archive is None:
            #     print "New product,",
            #     self.insert_and_fingerprint(prod)
            # else:  # means the item is in the archive
            #     # No matter what, we're moving this out of the archive...
            #     prod_in_archive["archive"] = False
            #     print "Prod in archive, checking fingerprint version...",
            #     if prod_in_archive.get("fp_version") == constants.fingerprint_version:
            #         print "fp_version good, moving from db.archive to db.products",
            #         prod_in_archive.update(prod)
            #         self.collection.insert_one(prod_in_archive)
            #     else:
            #         print "old fp_version,",
            #         self.insert_and_fingerprint(prod)
            #     self.db.archive.delete_one({'id': prod["id"]})
        else:
            # case 2: the product was found in our db, and maybe should be modified
            print "Found existing prod in db,",
            # Thus - update only shopstyle's fields
            if prod_in_products.get("lastModified", None) != prod.get("lastModified",
                                                                      None):  # assuming shopstyle update it correctly
                print "lastModifieds are different, updating SS fields",
                self.shopstyle_fields_update(prod)
                self.collection.update({'id': prod["id"]}, {'$set': {'dl_version': self.current_dl_version}})
                # This is now done at the beginning (by setting dl_version in prod ahead of time)
                # self.db.products.update({'id': prod["id"]}, {'$set': {'dl_version': self.current_dl_version}})


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


if __name__ == "__main__":
    update_db = ShopStyleDownloader()
    update_db.run_by_category()
