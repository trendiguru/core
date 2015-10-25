__author__ = 'yonatan'

"""
the download worker
"""

import collections
import time
import urllib

import requests
from rq import Queue

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
# collection = constants.update_collection_name
relevant = constants.db_relevant_items
fp_version = constants.fingerprint_version


def download_category(category_id):
    if category_id not in relevant:
        return
    category = db.categories.find_one({"id": category_id})
    if "count" in category and category["count"] <= MAX_SET_SIZE:
        print("Attempting to download: {0} products".format(category["count"]))
        print("Category: " + category_id)
        download_products({"pid": PID, "cat": category_id})
    elif "childrenIds" in category.keys():
        print("Splitting {0} products".format(category["count"]))
        print("Category: " + category_id)
        for child_id in category["childrenIds"]:
            download_category(child_id)
    else:
        initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": category["id"]})
        divide_and_conquer(initial_filter_params, 0)

        # self.archive_products(category_id)  # need to count how many where sent to archive


def divide_and_conquer(filter_params, filter_index):
    """Keep branching until we find disjoint subsets which have less then MAX_SET_SIZE items"""
    if filter_index >= len(FILTERS):
        # TODO: determine appropriate behavior in this case
        print "ran out of FILTERS"
        return
    filter_params["filters"] = FILTERS[filter_index]

    histogram_response = delayed_requests_get(BASE_URL_PRODUCTS + "histogram", filter_params)
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
        download_products(filter_params)
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
                download_products(subset_filter_params)
            else:
                print "Splitting: {0} products".format(subset["count"])
                print "Params: {0}".format(subset_filter_params.encoded())
                divide_and_conquer(subset_filter_params, filter_index + 1)


def download_products(filter_params, total=MAX_SET_SIZE):
    if not isinstance(filter_params, UrlParams):
        filter_params = UrlParams(params_dict=filter_params)

    # dl_query = {"filter_params": filter_params.encoded()}
    #
    # if self.db.dl_cache.find_one(dl_query):
    #     print "We've done this batch already, let's not repeat work"
    #     return

    if "filters" in filter_params:
        del filter_params["filters"]
    filter_params["limit"] = MAX_RESULTS_PER_PAGE
    filter_params["offset"] = 0
    while filter_params["offset"] < MAX_OFFSET and \
                    (filter_params["offset"] + MAX_RESULTS_PER_PAGE) <= total:
        product_response = delayed_requests_get(BASE_URL_PRODUCTS, filter_params)
        product_results = product_response.json()
        total = product_results["metadata"]["total"]
        products = product_results["products"]
        for prod in products:
            db_update(prod)
        filter_params["offset"] += MAX_RESULTS_PER_PAGE

        # Write down that we did this
        # self.db.dl_cache.insert(dl_query)

        # print "Batch Done. Total Product count: {0}".format(collection.count())


def delayed_requests_get(url, _params):
    sleep_time = max(0, 0.1 - (time.time() - db.DBD.find()[0]["last_request"]))
    time.sleep(sleep_time)
    db.DBD.find_one_and_update({"criteria": "main"},
                               {'$set': {"last_request": time.time()}})
    return requests.get(url, params=_params)


def db_update(prod):
    db.DBD.find_one_and_update({"criteria": "main"},
                               {'$inc': {"items_downloaded": 1}})
    return


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
