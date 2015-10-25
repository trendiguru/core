__author__ = 'yonatan'

"""
DB downloader
"""
import collections
import time
import urllib
import datetime

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
        self.last_request_time = time.time()
        self.do_fingerprint = True

    def run_by_category(self, do_fingerprint=True, type="DAILY"):
        x = raw_input("choose your update type - Daily or Full? (D/F)")
        if x is "f" or x is "F":
            type = "FULL"
        self.db.DBD.delete_many({})
        self.db.DBD.insert_one({"criteria": "main",
                                "last_request": self.last_request_time,
                                "items_downloaded": 0})
        self.do_fingerprint = do_fingerprint  # check if relevent
        root_category, ancestors = self.build_category_tree()
        cats_to_dl = [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            self.download_category(cat)

        print type + " DOWNLOAD DONE!!!!!\n"

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
            print "enqueuing for download",
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
            print "enqueuing for download",
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
                    print "enqueuing for download",
                    q.enqueue(download_products, filter_params=subset_filter_params)
                else:
                    print "Splitting: {0} products".format(subset["count"])
                    print "Params: {0}".format(subset_filter_params.encoded())
                    self.divide_and_conquer(subset_filter_params, filter_index + 1)

    def delayed_requests_get(self, url, _params):
        sleep_time = max(0, 0.1 - (time.time() - self.db.DBD.find()[0]["last_request"]))
        time.sleep(sleep_time)
        self.db.DBD.find_one_and_update({"criteria": "main"},
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
