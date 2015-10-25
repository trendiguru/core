import time
import datetime
import collections
import urllib

import requests
from rq import Queue

from DBDworker import download_products
import constants

q = Queue('DBD', connection=constants.redis_conn)

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
collection = constants.update_collection_name
relevant = constants.db_relevant_items
fp_version = constants.fingerprint_version

class ShopStyleDownloader():
    def __init__(self):
        # connect to db
        self.db = db
        self.collection = self.db[collection]
        self.current_dl_date = str(datetime.datetime.date(datetime.datetime.now()))
        self.do_fingerprint = True

    def run_by_category(self, do_fingerprint=True, type="DAILY"):
        x = raw_input("choose your update type - Daily or Full? (D/F)")
        if x is "f" or x is "F":
            type = "FULL"
        if self.db.download_data.find().count() < 1 or \
                        self.db.download_data.find()[0]["current_dl"] != self.current_dl_date or \
                type == "FULL":
            self.db.download_data.delete_many({})
            self.db.download_data.insert_one({"criteria": "main",
                                              "current_dl": self.current_dl_date,
                                              "start_time": datetime.datetime.now(),
                                              "items_downloaded": 0,
                                              "new_items": 0,
                                              "returned_from_archive": 0,
                                              "sent_to_archive": 0,
                                              "existing_items": 0,
                                              "existing_but_renewed": 0,
                                              "errors": 0,
                                              "end_time": "still in process",
                                              "total_dl_time": "still in process",
                                              "last_request": time.time()})
            self.db.drop_collection("fp_in_process")
            self.db.fp_in_process.insert_one({})
            self.db.fp_in_process.create_index("id")
            self.db.dl_cache.delete_many({})
            self.db.dl_cache.create_index("filter_params")
        # self.db.archive.create_index("id")
        self.do_fingerprint = do_fingerprint  # check if relevent
        root_category, ancestors = self.build_category_tree()
        cats_to_dl = [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            self.download_category(cat)
        self.wait_for(30)
        self.db.download_data.find_one_and_update({"criteria": "main"}, {'$set': {"end_time": datetime.datetime.now()}})
        tmp = self.db.download_data.find()[0]
        total_time = abs(tmp["end_time"] - tmp["start_time"]).total_seconds()
        self.db.download_data.find_one_and_update({"criteria": "main"},
                                                  {'$set': {"total_dl_time(hours)": str(total_time / 3600)[:5]}})
        del_items = self.collection.delete_many({'fingerprint': {"$exists": False}})
        print str(del_items.deleted_count) + ' items without fingerprint were deleted!\n'
        self.db.drop_collection("fp_in_process")
        print type + " DOWNLOAD DONE!!!!!\n"

    def wait_for(self, approx):
        x = raw_input("waitfor enabled? (Y/N)")
        if x == "n" or x == "N":
            return
        time.sleep(approx * 60)  # wait for the crolwer to download the data
        dl_data = self.db.download_data.find()[0]
        total_items = self.db.products.count()
        downloaded_items = dl_data["items_downloaded"]
        new_items = dl_data["new_items"]
        insert_errors = dl_data["errors"]
        sub = downloaded_items - insert_errors
        if total_items > sub:
            time.sleep(new_items / 100)
        else:
            check = 0
            while sub > total_items:
                if check > 24:
                    break
                print "\ncheck number " + str(check)
                print "\nfp workers didn't finish yet\nWaiting 5 min before checking again\n"
                check += 1
                print "check number" + str(check)
                time.sleep(300)
                total_items = db.products.count()
                insert_errors = dl_data["errors"]
                sub = downloaded_items - insert_errors

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
        if category_id not in relevant:
            return
        category = self.db.categories.find_one({"id": category_id})
        if "count" in category and category["count"] <= MAX_SET_SIZE:
            print("Attempting to download: {0} products".format(category["count"]))
            print("Category: " + category_id)
            q.enqueue(download_products, filter_params={"pid": PID, "cat": category_id})

        elif "childrenIds" in category.keys():
            print("Splitting {0} products".format(category["count"]))
            print("Category: " + category_id)
            for child_id in category["childrenIds"]:
                self.download_category(child_id)
        else:
            initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": category["id"]})
            self.divide_and_conquer(initial_filter_params, 0)
            # self.archive_products(category_id)  # need to count how many where sent to archive

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
            q.enqueue(download_products, filter_params=filter_params)
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
                    q.enqueue(download_products, filter_params=subset_filter_params)
                else:
                    print "Splitting: {0} products".format(subset["count"])
                    print "Params: {0}".format(subset_filter_params.encoded())
                    self.divide_and_conquer(subset_filter_params, filter_index + 1)

    def delayed_requests_get(self, url, _params):
        dl_data = self.db.download_data.find()[0]
        sleep_time = max(0, 0.1 - (time.time() - dl_data["last_request"]))
        time.sleep(sleep_time)
        self.db.download_data.find_one_and_update({"criteria": "main"},
                                                  {'$set': {"last_request": time.time()}})
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

if __name__ == "__main__":
    update_db = ShopStyleDownloader()
    update_db.run_by_category()

# import collections
# import time
# import json
# import urllib
# import datetime
#
# import requests
# from rq import Queue
#
# from fingerprint_core import generate_mask_and_insert
# import constants
#
# q = Queue('fingerprint_new', connection=constants.redis_conn)
#
# BASE_URL = "http://api.shopstyle.com/api/v2/"
# BASE_URL_PRODUCTS = BASE_URL + "products/"
# PID = "uid900-25284470-95"
# TG_FIELDS = ["fp_version", "fingerprint", "fp_date", "human_bb", "bounding_box"]
# # ideally, get filter list from DB. for now:
# # removed "Category" for now because its results  are not disjoint (sum of subsets > set)
# FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
# MAX_RESULTS_PER_PAGE = 50
# MAX_OFFSET = 5000
# MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE
# db = constants.db
# collection = constants.update_collection_name
# relevant = constants.db_relevant_items
# fp_version = constants.fingerprint_version
#
# class ShopStyleDownloader():
#     def __init__(self):
#         # connect to db
#         self.db = db
#         self.collection = self.db[collection]
#         self.current_dl_date = str(datetime.datetime.date(datetime.datetime.now()))
#         self.last_request_time = time.time()
#         self.do_fingerprint = True
#
#     def _run_by_filter(self):
#         # Let's try an initial category ("womens-suits")
#         initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": "womens-suits", "Sort": "Recency"})
#         self.divide_and_conquer(initial_filter_params, 0)
#
#     def run_by_category(self, do_fingerprint=True, type="DAILY"):
#         x = raw_input("choose your update type - Daily or Full? (D/F)")
#         if x is "f" or x is "F":
#             type = "FULL"
#         self.do_fingerprint = do_fingerprint  # check if relevent
#         root_category, ancestors = self.build_category_tree()
#         if self.db.download_data.find().count() < 1 or \
#                         self.db.download_data.find()[0]["current_dl"] != self.current_dl_date or \
#                 type == "FULL":
#             self.db.download_data.delete_many({})
#             self.db.download_data.insert_one({"criteria": "main",
#                                               "current_dl": self.current_dl_date,
#                                               "start_time": datetime.datetime.now(),
#                                               "items_downloaded": 0,
#                                               "new_items": 0,
#                                               "returned_from_archive": 0,
#                                               "sent_to_archive": 0,
#                                               "existing_items": 0,
#                                               "existing_but_renewed": 0,
#                                               "errors": 0,
#                                               "end_time": "still in process",
#                                               "total_dl_time": "still in process"})
#             self.db.drop_collection("fp_in_process")
#             self.db.fp_in_process.insert_one({})
#             self.db.fp_in_process.create_index("id")
#             self.db.dl_cache.delete_many({})
#             self.db.dl_cache.create_index("filter_params")
#         # self.db.archive.create_index("id")
#
#         cats_to_dl = [anc["id"] for anc in ancestors]
#         for cat in cats_to_dl:
#             self.download_category(cat)
#
#         self.db.download_data.find_one_and_update({"criteria": "main"}, {'$set': {"end_time": datetime.datetime.now()}})
#         tmp = self.db.download_data.find()[0]
#         total_time = abs(tmp["end_time"] - tmp["start_time"]).total_seconds()
#         self.db.download_data.find_one_and_update({"criteria": "main"},
#                                                   {'$set': {"total_dl_time(hours)": total_time / 3600}})
#         del_items = self.collection.delete_many({'fingerprint': {"$exists": False}})
#         print str(del_items.deleted_count) + ' items without fingerprint were deleted!\n'
#         self.db.drop_collection("fp_in_process")
#         print type + " DOWNLOAD DONE!!!!!\n"
#
#     def build_category_tree(self):
#         # download all categories
#         category_list_response = requests.get(BASE_URL + "categories", params={"pid": PID})
#         category_list_response_json = category_list_response.json()
#         root_category = category_list_response_json["metadata"]["root"]["id"]
#         category_list = category_list_response_json["categories"]
#         self.db.categories.remove({})
#         self.db.categories.insert(category_list)
#         # find all the children
#         for cat in self.db.categories.find():
#             self.db.categories.update({"id": cat["parentId"]}, {"$addToSet": {"childrenIds": cat["id"]}})
#         # get list of all categories under root - "ancestors"
#         ancestors = []
#         for c in self.db.categories.find({"parentId": root_category}):
#             ancestors.append(c)
#             # let's get some numbers in there - get a histogram for each ancestor
#         for anc in ancestors:
#             response = self.delayed_requests_get(BASE_URL_PRODUCTS + "histogram",
#                                                  {"pid": PID, "filters": "Category", "cat": anc["id"]})
#             hist = response.json()["categoryHistogram"]
#             # save count for each category
#             for cat in hist:
#                 self.db.categories.update({"id": cat["id"]}, {"$set": {"count": cat["count"]}})
#         return root_category, ancestors
#
#     def download_category(self, category_id):
#         if category_id not in relevant:
#             return
#         category = self.db.categories.find_one({"id": category_id})
#         if "count" in category and category["count"] <= MAX_SET_SIZE:
#             print("Attempting to download: {0} products".format(category["count"]))
#             print("Category: " + category_id)
#             self.download_products({"pid": PID, "cat": category_id})
#         elif "childrenIds" in category.keys():
#             print("Splitting {0} products".format(category["count"]))
#             print("Category: " + category_id)
#             for child_id in category["childrenIds"]:
#                 self.download_category(child_id)
#         else:
#             initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": category["id"]})
#             self.divide_and_conquer(initial_filter_params, 0)
#
#             # self.archive_products(category_id)  # need to count how many where sent to archive
#
#     def divide_and_conquer(self, filter_params, filter_index):
#         """Keep branching until we find disjoint subsets which have less then MAX_SET_SIZE items"""
#         if filter_index >= len(FILTERS):
#             # TODO: determine appropriate behavior in this case
#             print "ran out of FILTERS"
#             return
#         filter_params["filters"] = FILTERS[filter_index]
#
#         histogram_response = self.delayed_requests_get(BASE_URL_PRODUCTS + "histogram", filter_params)
#         # TODO: check the request worked here
#         try:
#             histogram_results = histogram_response.json()
#             # for some reason Category doesn't come with a prefix...
#
#             filter_prefix = histogram_results["metadata"].get("histograms")[0].get("prefix")
#
#             hist_key = FILTERS[filter_index].lower() + "Histogram"
#             if hist_key not in histogram_results:
#                 raise RuntimeError("No hist_key {0} in histogram: \n{1}".format(hist_key, histogram_results))
#         except:
#             print "Could not get histogram for filter_params: {0}".format(filter_params.encoded())
#             print "Attempting to downloading first 5000"
#             # TODO: Better solution for this
#             self.download_products(filter_params)
#         else:
#             hist_key = FILTERS[filter_index].lower() + "Histogram"
#             for subset in histogram_results[hist_key]:
#                 subset_filter_params = filter_params.copy()
#                 if FILTERS[filter_index] == "Category":
#                     _key = "cat"
#                     filter_prefix = ""
#                 else:
#                     _key = "fl"
#                 subset_filter_params[_key] = filter_prefix + subset["id"]
#                 if subset["count"] < MAX_SET_SIZE:
#                     self.download_products(subset_filter_params)
#                 else:
#                     print "Splitting: {0} products".format(subset["count"])
#                     print "Params: {0}".format(subset_filter_params.encoded())
#                     self.divide_and_conquer(subset_filter_params, filter_index + 1)
#
#     def download_products(self, filter_params, total=MAX_SET_SIZE):
#         if not isinstance(filter_params, UrlParams):
#             filter_params = UrlParams(params_dict=filter_params)
#
#         dl_query = {"filter_params": filter_params.encoded()}
#
#         if self.db.dl_cache.find_one(dl_query):
#             print "We've done this batch already, let's not repeat work"
#             return
#
#         if "filters" in filter_params:
#             del filter_params["filters"]
#         filter_params["limit"] = MAX_RESULTS_PER_PAGE
#         filter_params["offset"] = 0
#         while filter_params["offset"] < MAX_OFFSET and \
#                         (filter_params["offset"] + MAX_RESULTS_PER_PAGE) <= total:
#             product_response = self.delayed_requests_get(BASE_URL_PRODUCTS, filter_params)
#             product_results = product_response.json()
#             total = product_results["metadata"]["total"]
#             products = product_results["products"]
#             for prod in products:
#                 self.db_update(prod)
#             filter_params["offset"] += MAX_RESULTS_PER_PAGE
#
#         # Write down that we did this
#         self.db.dl_cache.insert(dl_query)
#
#         print "Batch Done. Total Product count: {0}".format(self.collection.count())
#
#     def archive_products(self, category_id=None):
#         print "archiving old products"
#         # this process iterate our whole DB (after daily download) and shifts unwanted items to the archive collection
#         old_items = self.collection.find({'download_data.dl_version': {'$not': self.current_dl_date}})
#         """
#         needs to write again
#
#         """
#         # query_doc = {'$and': [
#         #     {'download_data': {'dl_version': {'$exists': 1, '$lt': self.current_dl}}},
#         #     {'$or': [
#         #         {'archive': {'$exists': 0}},
#         #         {'archive': False}
#         #     ]}]}
#         #
#         # if category_id:
#         #     query_doc["$and"].append(
#         #         {"categories":
#         #             {"$elemMatch": {
#         #                 "id": {"$in": find_similar_mongo.get_all_subcategories(self.db.categories, category_id)}}}
#         #         })
#
#         # update_result = self.collection.update_many(query_doc, {"$set": {"archive": True}})
#         # print "Marked {0} of {1} products for archival".format(update_result.modified_count,
#         #                                                        update_result.matched_count)
#
#     def delayed_requests_get(self, url, _params):
#         sleep_time = max(0, 0.1 - (time.time() - self.last_request_time))
#         time.sleep(sleep_time)
#         self.last_request_time = time.time()
#         return requests.get(url, params=_params)
#
#     def shopstyle_fields_update(self, prod):
#         """
#         this func. updates only shopstyle's fields of a product in the DB
#         :param prod: dictionary of a DB product
#         :return: Nothing, void function
#         """
#         self.collection.update_one({'id': prod['id']}, {'$set': prod})
#
#     def insert_and_fingerprint(self, prod, do_fingerprint=None):
#         """
#         this func. inserts a new product to our DB and runs TG fingerprint on it
#         :param prod: dictionary of shopstyle product
#         :return: Nothing, void function
#         """
#         if do_fingerprint is None:
#             do_fingerprint = self.do_fingerprint
#
#         if do_fingerprint:
#             print "enqueuing for fingerprint & insert,",
#             q.enqueue(generate_mask_and_insert, doc=prod, image_url=None, mask_only=False,
#                       fp_date=self.current_dl_date)
#
#     def db_update(self, prod):
#         print ""
#         print "Updating product {0}. ".format(prod["id"]),
#
#         # requests package can't handle https - temp fix
#         prod["image"] = json.loads(json.dumps(prod["image"]).replace("https://", "http://"))
#         prod_in_que = self.db.fp_in_process.find_one({"id": prod["id"]})
#         if prod_in_que is not None:
#             return
#         self.db.download_data.find_one_and_update({"criteria": "main"},
#                                                   {'$inc': {"items_downloaded": 1}})
#         prod["download_data"] = {"dl_version": self.current_dl_date}
#         # case 1: new product - try to update, if does not exists, insert a new product and add our fields
#         prod_in_products = self.collection.find_one({"id": prod["id"]})
#         category = prod['categories'][0]['id']
#         print category
#         if prod_in_products is None:
#             print "Product not in db.products, searching in archive. ",
#             # case 1.1: try finding this product in the archive
#             prod_in_archive = self.db.archive.find_one({'id': prod["id"]})
#             if prod_in_archive is None:
#                 print "New product,",
#                 self.db.download_data.find_one_and_update({"criteria": "main"},
#                                                           {'$inc': {"new_items": 1}})
#                 self.db.fp_in_process.insert_one({"id": prod["id"]})
#                 self.insert_and_fingerprint(prod)
#             else:  # means the item is in the archive
#                 # No matter what, we're moving this out of the archive...
#                 prod_in_archive["archive"] = False
#                 print "Prod in archive, checking fingerprint version...",
#                 if prod_in_archive.get("download_data")["fp_version"] == constants.fingerprint_version:
#                     print "fp_version good, moving from db.archive to db.products",
#                     prod_in_archive.update(prod)
#                     self.collection.insert_one(prod_in_archive)
#                 else:
#                     print "old fp_version, updating fp",
#                     self.insert_and_fingerprint(prod)
#                 self.db.archive.delete_one({'id': prod["id"]})
#                 self.db.download_data.find_one_and_update({"criteria": "main"},
#                                                           {'$inc': {"returned_from_archive": 1}})
#         else:
#             # case 2: the product was found in our db, and maybe should be modified
#             print "Found existing prod in db,",
#             if prod_in_products.get("download_data")["fp_version"] == fp_version:
#                 # Thus - update only shopstyle's fields
#                 self.db.download_data.find_one_and_update({"criteria": "main"},
#                                                           {'$inc': {"existing_items": 1}})
#                 self.collection.update_one({'id': prod["id"]},
#                                            {'$set': {'download_data.dl_version': self.current_dl_date}})
#             else:
#                 self.db.download_data.find_one_and_update({"criteria": "main"},
#                                                           {'$inc': {"existing_but_renewed": 1}})
#                 self.collection.delete_one({'id': prod['id']})
#                 self.insert_and_fingerprint(prod)
#                 print "product with an old fp was refingerprinted"
#
#
# class UrlParams(collections.MutableMapping):
#     """This is current designed specifically for the ShopStyle API where there can be multiple "fl" parameters,
#     although it could easily be made more modular, where a list of keys which can have multiple values is predefined,
#     or the option (bool multivalue) is supplied on new entry creation...
#     """
#
#     def __init__(self, params_dict={}, _fl_list=[]):
#         self.p_dict = dict(params_dict)
#         self.fl_list = list(_fl_list)
#
#     def __iter__(self):
#         for key in self.p_dict:
#             yield key
#         for i in range(0, len(self.fl_list)):
#             yield ("fl", i)
#
#     def __getitem__(self, key):
#         if isinstance(key, tuple):
#             return self.fl_list[key[1]]
#         elif key == "fl":
#             return self.fl_list
#         else:
#             return self.p_dict[key]
#
#     def __setitem__(self, key, value):
#         if key == "fl":
#             self.fl_list.append(value)
#         else:
#             self.p_dict[key] = value
#
#     def __delitem__(self, key):
#         del self.p_dict[key]
#
#     def __len__(self):
#         return len(self.p_dict) + len(self.fl_list)
#
#     def iteritems(self):
#         for k, v in self.p_dict.iteritems():
#             yield k, v
#         for item in self.fl_list:
#             yield 'fl', item
#
#     def items(self):
#         return [(k, v) for k, v in self.iteritems()]
#
#     def copy(self):
#         return UrlParams(self.p_dict, self.fl_list)
#
#     @staticmethod
#     def encode_params(data):
#         """Encode parameters in a piece of data.
#         Will successfully encode parameters when passed as a dict or a list of
#         2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
#         if parameters are supplied as a dict.
#
#         Taken from requests package:
#         https://github.com/kennethreitz/requests/blob/8b5e457b756b2ab4c02473f7a42c2e0201ecc7e9/requests/models.py
#         """
#
#         def to_key_val_list(value):
#             """Take an object and test to see if it can be represented as a
#             dictionary. If it can be, return a list of tuples, e.g.,
#             ::
#                 >>> to_key_val_list([('key', 'val')])
#                 [('key', 'val')]
#                 >>> to_key_val_list({'key': 'val'})
#                 [('key', 'val')]
#                 >>> to_key_val_list('string')
#                 ValueError: cannot encode objects that are not 2-tuples.
#             """
#             if value is None:
#                 return None
#
#             if isinstance(value, (str, bytes, bool, int)):
#                 raise ValueError('cannot encode objects that are not 2-tuples')
#
#             if isinstance(value, collections.Mapping):
#                 value = value.items()
#
#             return list(value)
#
#         if isinstance(data, (str, bytes)):
#             return data
#         elif hasattr(data, 'read'):
#             return data
#         elif hasattr(data, '__iter__'):
#             result = []
#             for k, vs in to_key_val_list(data):
#                 if isinstance(vs, basestring) or not hasattr(vs, '__iter__'):
#                     vs = [vs]
#                 for v in vs:
#                     if v is not None:
#                         result.append(
#                             (k.encode('utf-8') if isinstance(k, str) else k,
#                              v.encode('utf-8') if isinstance(v, str) else v))
#             return urllib.urlencode(result, doseq=True)
#         else:
#             return data
#
#     def encoded(self):
#         return self.__class__.encode_params(self)
#
#
# if __name__ == "__main__":
#     update_db = ShopStyleDownloader()
#     update_db.run_by_category()
