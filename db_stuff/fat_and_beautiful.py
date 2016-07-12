"""
126. plus-sizes
127. plus-size-jeans
128. plus-size-dresses
129. plus-size-intimates
130. plus-size-jackets
131. plus-size-outerwear
132. plus-size-pants
133. plus-size-shorts
134. plus-size-skirts
135. plus-size-sweatshirts
136. plus-size-sweaters
137. plus-size-swimsuits
138. plus-size-tops
343. mens-big-and-tall
344. mens-big-and-tall-coats-and-jackets
345. mens-big-and-tall-jeans
346. mens-big-and-tall-pants
347. mens-big-and-tall-shirts
348. mens-big-and-tall-shorts
349. mens-big-and-tall-blazers
350. mens-big-and-tall-suits
351. mens-big-and-tall-sweaters
"""


__author__ = 'yonatan'

import collections
import time
import json
import urllib
import datetime
import sys
import argparse
import requests
from rq import Queue
from fanni import plantForests4AllCategories
from ..constants import db, fingerprint_version as fp_version, redis_conn
from . import shopstyle_constants
from .shopstyle2generic import convert2generic
from ..fingerprint_core import generate_mask_and_insert
from . import dl_excel


q = Queue('fingerprint_new', connection=redis_conn)
forest = Queue('annoy_forest', connection=redis_conn)

BASE_URL = "http://api.shopstyle.com/api/v2/"
BASE_URL_PRODUCTS = BASE_URL + "products/"
PID = "uid900-25284470-95"
# ideally, get filter list from DB. for now:
# removed "Category" for now because its results  are not disjoint (sum of subsets > set)
FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
MAX_RESULTS_PER_PAGE = 50
MAX_OFFSET = 5000
MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE


class ShopStyleDownloader:

    def __init__(self, collection, gender):
        dl_cache = collection +'_cache'
        self.db = db
        self.collection_name = collection
        self.collection = self.db[collection]
        self.collection_archive = self.db[collection+'_archive']
        self.cache = self.db[dl_cache]
        self.cache_counter = 0
        self.categories = self.db.categories
        self.current_dl_date = str(datetime.datetime.date(datetime.datetime.now()))
        self.last_request_time = time.time()
        if gender == 'Female':
            self.gender = 'Female'
            self.relevant = shopstyle_constants.fat2paperdoll_Female.keys()
        else:
            self.gender = 'Male'
            self.relevant = shopstyle_constants.fat2paperdoll_Male.keys()
        self.status = self.db.download_status
        self.status_full_path = "collections." + self.collection_name + ".status"
        self.notes_full_path = "collections." + self.collection_name + ".notes"
        self.status.update_one({"date": self.current_dl_date}, {"$set": {self.status_full_path: "Working"}})

    def db_download(self):
        start_time = time.time()

        self.cache.create_index("filter_params")
        self.cache.create_index("dl_version")
        ancestors = self.categories.find()

        cats_to_dl = [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            self.download_category(cat)

        self.wait_for_fingerprint_q_to_be_empty()
        end_time= time.time()
        total_time = (end_time - start_time)/3600
        self.status.update_one({"date": self.current_dl_date}, {"$set": {self.status_full_path: "Finishing Up"}})
        del_items = self.collection.delete_many({'fingerprint': {"$exists": False}})
        self.theArchiveDoorman()

        dl_info = {"date": self.current_dl_date,
           "dl_duration": total_time,
           "store_info": []}

        self.cache.delete_many({})
        dl_excel.mongo2xl(self.collection_name, dl_info)
        print self.collection_name + " DOWNLOAD DONE!!!!!\n"
        new_items = self.collection.find({'download_data.first_dl': self.current_dl_date}).count()
        self.status.update_one({"date": self.current_dl_date}, {"$set": {self.status_full_path: "Done",
                                                                         self.notes_full_path: new_items}})

    def theArchiveDoorman(self):
        # clean the archive from items older than a week
        archivers = self.collection_archive.find()
        y_new, m_new, d_new = map(int, self.current_dl_date.split("-"))
        for item in archivers:
            y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
            days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
            if days_out < 7:
                self.collection_archive.update_one({'id': item['id']}, {"$set": {"status.days_out": days_out}})
            else:
                self.collection_archive.delete_one({'id': item['id']})

        # add to the archive items which were not downloaded today but were instock yesterday
        notUpdated = self.collection.find({"download_data.dl_version": {"$ne": self.current_dl_date}})
        for item in notUpdated:
            self.collection.delete_one({'id': item['id']})
            existing = self.collection_archive.find_one({"id": item["id"]})
            if existing:
                continue
            y_old, m_old, d_old = map(int, item["download_data"]["dl_version"].split("-"))
            days_out = 365 * (y_new - y_old) + 30 * (m_new - m_old) + (d_new - d_old)
            if days_out < 7:
                item['status']['instock'] = False
                item['status']['days_out'] = days_out
                self.collection_archive.insert_one(item)


        # move to the archive all the items which were downloaded today but are out of stock
        outStockers = self.collection.find({'status.instock': False})
        for item in outStockers:
            self.collection.delete_one({'id':item['id']})
            existing = self.collection_archive.find_one({"id": item["id"]})
            if existing:
                continue
            self.collection_archive.insert_one(item)

        self.collection_archive.reindex()

    def wait_for_fingerprint_q_to_be_empty(self):
        check = 0
        while q.count>1:
            if check > 36:
                break
            check += 1
            time.sleep(300)

    def download_category(self, category_id):
        if category_id not in self.relevant:
            return
        parameters = {"pid": PID}

        category = self.categories.find_one({"id": category_id})
        parameters["cat"] = category_id
        if "count" in category and category["count"] <= MAX_SET_SIZE:
            print("Attempting to download: {0} products".format(category["count"]))
            self.download_products(parameters)
        elif "childrenIds" in category.keys():
            print("Category: " + category_id)
            for child_id in category["childrenIds"]:
                self.download_category(child_id)
        else:
            initial_filter_params = UrlParams(params_dict=parameters)
            self.divide_and_conquer(initial_filter_params, 0)

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
                    self.download_products(subset_filter_params)
                else:
                    # print "Splitting: {0} products".format(subset["count"])
                    # print "Params: {0}".format(subset_filter_params.encoded())
                    self.divide_and_conquer(subset_filter_params, filter_index + 1)

    def download_products(self, filter_params, total=MAX_SET_SIZE):
        """
        Download with paging...
        :param filter_params:
        :param total:
        """
        if not isinstance(filter_params, UrlParams):
            filter_params = UrlParams(params_dict=filter_params)

        dl_query = {"dl_version": self.current_dl_date,
                    "filter_params": filter_params.encoded()}

        if self.cache.find_one(dl_query):
            self.cache_counter +=1
            print "%s ) We've done this batch already, let's not repeat work" %str(self.cache_counter)
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
                    self.db_update(prod)
                filter_params["offset"] += MAX_RESULTS_PER_PAGE

        # Write down that we did this
        self.cache.insert(dl_query)

        # print "Batch Done. Total Product count: {0}".format(self.collection.count())

    def delayed_requests_get(self, url, params):
        sleep_time = max(0, 0.1 - (time.time() - self.last_request_time))
        time.sleep(sleep_time)
        self.last_request_time = time.time()
        return requests.get(url, params=params)

    def insert_and_fingerprint(self, prod):
        """
        this func. inserts a new product to our DB and runs TG fingerprint on it
        :param prod: dictionary of shopstyle product
        :return: Nothing, void function
        """

        while q.count>250000:
            print ("Q full - stolling")
            time.sleep(600)

        q.enqueue(generate_mask_and_insert, doc=prod, image_url=prod["images"]["XLarge"],
                  fp_date=self.current_dl_date, coll=self.collection_name)
        # print "inserting,",

    def db_update(self, prod):
        # print "";print "Updating product {0}. ".format(prod["id"]),

        # requests package can't handle https - temp fix
        prod["image"] = json.loads(json.dumps(prod["image"]).replace("https://", "http://"))
        prod["download_data"] = {"dl_version": self.current_dl_date}

        # case 1: new product - try to update, if does not exists, insert a new product and add our fields
        prod_in_coll = self.collection.find_one({"id": prod["id"]})

        if prod_in_coll is None:
            # print "Product not in db." + collection
            # case 1.1: try finding this product in the products
            if self.collection_name in ['GangnamStyle_Female','GangnamStyle_Male'] :
                if self.gender =='Female':
                    prod_in_prod = self.db.ShopStyle_Female.find_one({"id": prod["id"]})
                else:
                    prod_in_prod = self.db.ShopStyle_Male.find_one({"id": prod["id"]})

                if prod_in_prod is not None:
                    # print "but new product is already in db.products"
                    prod["download_data"] = prod_in_prod["download_data"]
                    prod = convert2generic(prod, self.gender)
                    prod["fingerprint"] = prod_in_prod["fingerprint"]
                    prod["download_data"]["dl_version"] = self.current_dl_date
                    self.collection.insert_one(prod)
                    return

            prod = convert2generic(prod, self.gender)
            if prod is None:
                return
            self.insert_and_fingerprint(prod)

        else:
            # case 2: the product was found in our db, and maybe should be modified
            # print "Found existing prod in db,",
            # Thus - update only shopstyle's fields
            status_new = prod["inStock"]
            status_old = prod_in_coll["status"]["instock"]
            if status_new is False and status_old is False:
                self.collection.update_one({'id': prod["id"]},
                                               {'$inc': {'status.days_out': 1}})
                prod["status"]["days_out"] = prod_in_coll["status"]["days"] + 1
            elif status_new is True and status_old is False:
                self.collection.update_one({'id': prod["id"]},
                                               {'$set': {'status.days_out': 0,
                                                         'status.instock': True}})
            else:
                pass

            if prod_in_coll["download_data"]["fp_version"] == fp_version:
                self.collection.update_one({'id': prod["id"]},
                                               {'$set': {'download_data.dl_version': self.current_dl_date}})
            else:
                self.collection.delete_one({'id': prod['id']})
                prod = convert2generic(prod, self.gender)
                self.insert_and_fingerprint(prod)


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

def getUserInput():
    parser = argparse.ArgumentParser(description='"@@@ Fat&Beauty Download @@@')
    parser.add_argument('-n', '--name',default="Fat_Beauty", dest= "name",
                        help='collection name')
    parser.add_argument('-g', '--gender', dest= "gender",
                        help='specify which gender to download. (Female or Male - case sensitive)', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    user_input = getUserInput()
    col = user_input.name
    gender = user_input.gender

    if gender in ['Female','Male'] :
        col = col + "_" +gender
    else:
        print("bad input - gender should be only Female or Male (case sensitive)")
        sys.exit(1)

    print ("@@@ ShopStyle Download @@@\n you choose to update the " + col + " collection")
    update_db = ShopStyleDownloader(col,gender)
    update_db.db_download()
    forest_job = forest.enqueue(plantForests4AllCategories, col_name=col, timeout=3600)
    while not forest_job.is_finished and not forest_job.is_failed:
        time.sleep(300)
    if forest_job.is_failed:
        print ('annoy plant forest failed')

    print (col + "Update Finished!!!")


