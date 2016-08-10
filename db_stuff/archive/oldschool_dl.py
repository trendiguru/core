__author__ = 'yonatan'

import collections
import time
import json
import urllib
import datetime
import sys

import requests
from rq import Queue

from core import constants
from core.db_stuff.shopstyle_constants import shopstyle_relevant_items_Female
from core.db_stuff.shopstyle2generic import convert2generic
from core.fingerprint_core import generate_mask_and_insert
from core.db_stuff import dl_excel

q = Queue('fingerprint_new', connection=constants.redis_conn)
relevant = shopstyle_relevant_items_Female

BASE_URL = "http://api.shopstyle.com/api/v2/"
BASE_URL_PRODUCTS = BASE_URL + "products/"
PID = "uid900-25284470-95"
# ideally, get filter list from DB. for now:
# removed "Category" for now because its results  are not disjoint (sum of subsets > set)
FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
MAX_RESULTS_PER_PAGE = 50
MAX_OFFSET = 5000
MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE
fp_version = constants.fingerprint_version


class ShopStyleDownloader():
    def __init__(self):
        # connect to db
        self.db = constants.db
        self.current_dl_date = str(datetime.datetime.date(datetime.datetime.now()))
        self.last_request_time = time.time()

    def db_download(self, collection):
        # if self.db.download_data.find({"criteria": collection}).count() > 0:
        #     self.db.download_data.delete_one({"criteria": collection})
        # self.db.download_data.insert_one({"criteria": collection,
        #                                   "current_dl": self.current_dl_date,
        #                                   "start_time": datetime.datetime.now(),
        #                                   "items_downloaded": 0,
        #                                   "new_items": 0,
        #                                   "errors": 0,
        #                                   "end_time": "still in process",
        #                                   "total_dl_time(min)": "still in process",
        #                                   "last_request": time.time(),
        #                                   "total_items": 0,
        #                                   "instock": 0,
        #                                   "out": 0})
        start_time = time.time()
        self.db.dl_cache.delete_many({})
        self.db.dl_cache.create_index("filter_params")
        root_category, ancestors = self.build_category_tree(collection)

        cats_to_dl = [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            self.download_category(cat, collection)

        self.wait_for(collection)
        end_time= time.time()
        # self.db.download_data.update_one({"criteria": collection},
        #                                           {'$set': {"end_time": datetime.datetime.now()}})
        # tmp = self.db.download_data.find({"criteria": collection})[0]
        # total_time = abs(tmp["end_time"] - tmp["start_time"]).total_seconds()
        total_time = (end_time - start_time)/3600
        del_items = self.db[collection].delete_many({'fingerprint': {"$exists": False}})
        # print str(del_items.deleted_count) + ' items without fingerprint were deleted!\n'
        # total_items = self.db[collection].count()
        old = self.db[collection].find({"download_data.dl_version": {"$ne": self.current_dl_date}})
        y_new, m_new, d_new = map(int, self.current_dl_date.split("-"))
        for item in old:
            y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
            days_out = 365*(y_new-y_old)+30*(m_new-m_old)+(d_new-d_old)
            self.db[collection].update_one({'id': item['id']}, {"$set": {"status.days_out": days_out,
                                                                         "status.instock": False}})
        #
        # instock = self.db[collection].find({"status.instock": True}).count()
        # out = self.db[collection].find({"status.instock": False}).count()
        # self.db.download_data.update_one({"criteria": collection},
        #                                           {'$set': {"total_dl_time(min)": str(total_time / 60)[:5],
        #                                                     "end_time": datetime.datetime.now(),
        #                                                     "total_items": str(total_items),
        #                                                     "instock": str(instock),
        #                                                     "out": str(out)}})
        dl_info = {"date": self.current_dl_date,
           "dl_duration": total_time,
           "store_info": []}

        dl_excel.mongo2xl('shopstyle', dl_info)
        print collection + " DOWNLOAD DONE!!!!!\n"


    def wait_for(self, collection):
        # print "Waiting for 45 min before first check"
        # total_items_before = self.db[collection].count()
        # time.sleep(2700)
        # total_items_after = self.db[collection].count()
        check = 0
        # checking if there is still change in the total items count
        # while total_items_before != total_items_after:
        while q.count>1:
            if check > 36:
                break
            # print "\ncheck number " + str(check)
            # print "\nfp workers didn't finish yet\nWaiting 5 min before checking again\n"
            check += 1
            time.sleep(300)
            # total_items_before = total_items_after
            # total_items_after = self.db[collection].count()

    def build_category_tree(self, collection):
        parameters = {"pid": PID, "filters": "Category"}
        if collection == "products_jp" or collection == "new_products_jp":
            parameters["site"] = "www.shopstyle.co.jp"

        # download all categories
        category_list_response = requests.get(BASE_URL + "categories", params=parameters)
        category_list_response_json = category_list_response.json()
        root_category = category_list_response_json["metadata"]["root"]["id"]
        category_list = category_list_response_json["categories"]
        self.db.categories.remove({})
        self.db.categories.insert(category_list)
        # find all the children

        for cat in self.db.categories.find():
            self.db.categories.update_one({"id": cat["parentId"]}, {"$addToSet": {"childrenIds": cat["id"]}})
        # get list of all categories under root - "ancestors"
        ancestors = []
        for c in self.db.categories.find({"parentId": root_category}):
            ancestors.append(c)
        # let's get some numbers in there - get a histogram for each ancestor
        for anc in ancestors:
            parameters["cat"] = anc["id"]
            response = self.delayed_requests_get(BASE_URL_PRODUCTS + "histogram", parameters)
            hist = response.json()["categoryHistogram"]
            # save count for each category
            for cat in hist:
                self.db.categories.update_one({"id": cat["id"]}, {"$set": {"count": cat["count"]}})
        return root_category, ancestors

    def download_category(self, category_id, collection):
        if category_id not in relevant:
            return
        parameters = {"pid": PID}  # , "filters": "Category"}
        if collection == "products_jp" or collection == "new_products_jp":
            parameters["site"] = "www.shopstyle.co.jp"
        category = self.db.categories.find_one({"id": category_id})
        parameters["cat"] = category["id"]
        if "count" in category and category["count"] <= MAX_SET_SIZE:
            print("Attempting to download: {0} products".format(category["count"]))
            #print("Category: " + category_id)
            self.download_products(parameters, coll=collection)
        elif "childrenIds" in category.keys():
            #print("Splitting {0} products".format(category["count"]))
            print("Category: " + category_id)
            for child_id in category["childrenIds"]:
                self.download_category(child_id, collection)
        else:
            initial_filter_params = UrlParams(params_dict=parameters)
            self.divide_and_conquer(initial_filter_params, 0, collection)


    def divide_and_conquer(self, filter_params, filter_index, coll="products"):
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
                    self.download_products(subset_filter_params, coll=coll)
                else:
                    # print "Splitting: {0} products".format(subset["count"])
                    # print "Params: {0}".format(subset_filter_params.encoded())
                    self.divide_and_conquer(subset_filter_params, filter_index + 1)

    def download_products(self, filter_params, total=MAX_SET_SIZE, coll="products"):
        """
        Download with paging...
        :param filter_params:
        :param total:
        """
        if not isinstance(filter_params, UrlParams):
            filter_params = UrlParams(params_dict=filter_params)

        dl_query = {"dl_version": self.current_dl_date,
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
            if product_response.status_code == 200:
                product_results = product_response.json()
                total = product_results["metadata"]["total"]
                products = product_results["products"]
                for prod in products:
                    self.db_update(prod, coll)
                filter_params["offset"] += MAX_RESULTS_PER_PAGE

        # Write down that we did this
        self.db.dl_cache.insert(dl_query)

        # print "Batch Done. Total Product count: {0}".format(self.db[coll].count())

    def delayed_requests_get(self, url, params):
        sleep_time = max(0, 0.1 - (time.time() - self.last_request_time))
        time.sleep(sleep_time)
        self.last_request_time = time.time()
        return requests.get(url, params=params)

    def insert_and_fingerprint(self, prod, collection):
        """
        this func. inserts a new product to our DB and runs TG fingerprint on it
        :param prod: dictionary of shopstyle product
        :return: Nothing, void function
        """
        while q.count>250000:
            print ("Q full - stolling")
            time.sleep(600)

        q.enqueue(generate_mask_and_insert, doc=prod, image_url=prod["images"]["XLarge"],
                  fp_date=self.current_dl_date, coll=collection)
        # print "inserting,",

    def db_update(self, prod, collection):
        # print ""
        # print "Updating product {0}. ".format(prod["id"]),

        # requests package can't handle https - temp fix
        prod["image"] = json.loads(json.dumps(prod["image"]).replace("https://", "http://"))
        # self.db.download_data.update_one({"criteria": collection},
        #                                           {'$inc': {"items_downloaded": 1}})
        prod["download_data"] = {"dl_version": self.current_dl_date}

        # case 1: new product - try to update, if does not exists, insert a new product and add our fields
        prod_in_coll = self.db[collection].find_one({"id": prod["id"]})

        if prod_in_coll is None:
            # print "Product not in db." + collection
            # case 1.1: try finding this product in the products
            if collection != "products":
                prod_in_prod = self.db.products.find_one({"id": prod["id"]})
                if prod_in_prod is not None:
                    # print "but new product is already in db.products"
                    prod["download_data"] = prod_in_prod["download_data"]
                    prod = convert2generic(prod)
                    prod["fingerprint"] = prod_in_prod["fingerprint"]
                    prod["download_data"]["dl_version"] = self.current_dl_date
                    self.db[collection].insert_one(prod)
                    return
            # self.db.download_data.update_one({"criteria": collection},
            #                                           {'$inc': {"new_items": 1}})
            prod = convert2generic(prod)
            self.insert_and_fingerprint(prod, collection)

        else:
            # case 2: the product was found in our db, and maybe should be modified
            # print "Found existing prod in db,",
            # Thus - update only shopstyle's fields
            status_new = prod["inStock"]
            status_old = prod_in_coll["status"]["instock"]
            if status_new is False and status_old is False:
                self.db[collection].update_one({'id': prod["id"]},
                                               {'$inc': {'status.days_out': 1}})
                prod["status"]["days_out"] = prod_in_coll["status"]["days"] + 1
            elif status_new is True and status_old is False:
                self.db[collection].update_one({'id': prod["id"]},
                                               {'$set': {'status.days_out': 0,
                                                         'status.instock': True}})
            else:
                pass

            if prod_in_coll["download_data"]["fp_version"] == fp_version:
                self.db[collection].update_one({'id': prod["id"]},
                                               {'$set': {'download_data.dl_version': self.current_dl_date}})
            else:
                self.db[collection].delete_one({'id': prod['id']})
                prod = convert2generic(prod)
                self.insert_and_fingerprint(prod, collection)


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
    col = "products"
    if len(sys.argv) == 2:
        col = col + "_" + sys.argv[1]
    print ("@@@ Shopstyle Download @@@\n you choose to update the " + col + " collection")
    update_db = ShopStyleDownloader()
    update_db.db_download(col)

    print (col + "Update Finished!!!")
