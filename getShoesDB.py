__author__ = 'Nadav Paz'

import collections
import time

import pymongo
import requests

import constants
import db_fingerprint_nadav


BASE_URL = "http://api.shopstyle.com/api/v2/"
BASE_URL_PRODUCTS = BASE_URL + "shoes/"
PID = "uid900-25284470-95"
TG_FIELDS = ["fp_version", "fingerprint", "fp_date", "human_bb", "bounding_box"]
# ideally, get filter list from DB. for now:
# removed "Category" for now because its results  are not disjoint (sum of subsets > set)
FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
MAX_RESULTS_PER_PAGE = 50
MAX_OFFSET = 5000
MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE


class ShopStyleDownloader():
    def __init__(self):
        # connect to db
#        self.db = pymongo.MongoClient('mongodb1-instance-1').mydb
        self.db = constants.db

    def run_by_filter(self):
        # Let's try an initial category ("womens-suits")
        initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": "womens-suits", "Sort": "Recency"})
        self.divide_and_conquer(initial_filter_params, 0)

    def run_by_category(self, root=None):
        root_category, ancestors = self.build_category_tree()
        # special case when we want to DL everything, because the root_category doesn't exit.
        if root is None:
            for anc in ancestors:
                self.download_category(anc["id"])
        else:
            self.download_category(root)
        archive_size_before_update = self.db.archive.count()
        self.archive_products()
        print("Done. Total products moved to archive: %i" % self.db.archive.count() - archive_size_before_update)

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
        if category["count"] <= MAX_SET_SIZE:
            print("Attempting to download: %i products" % category["count"])
            print("Category: " + category_id)
            self.download_products({"pid": PID, "cat": category_id})
        elif "childrenIds" in category.keys():
            print("Splitting %i products" % category["count"])
            print("Category: " + category_id)
            for child_id in category["childrenIds"]:
                self.download_category(child_id)
        else:
            initial_filter_params = UrlParams(params_dict={"pid": PID, "cat": category["id"]})
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
        histogram_results = histogram_response.json()
        # for some reason Category doesn't come with a prefix...

        filter_prefix = histogram_results["metadata"].get("histograms")[0].get("prefix")
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
                print("Splitting: %i products" % subset["count"])
                print("Params: " + str(subset_filter_params.items()))
                self.divide_and_conquer(subset_filter_params, filter_index + 1)

    def download_products(self, filter_params, total=MAX_SET_SIZE):
        """

        :param filter_params:
        :param total:
        """
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
        print("Done. Total Product count: %i" % self.db.shoes.count())

    def archive_products(self):
        # this process iterate our whole DB (after daily download) and shifts unwanted items to the archive collection
        current_version = self.db.globals.find_one({'dl_version': {'$exists': 1}})['dl_version']
        for prod in self.db.shoes.find():
            if prod['dl_version'] is not current_version:
                self.db.archive.insert(prod)
                self.db.shoes.remove({'id': prod['id']})
        self.db.globals.update({'dl_version': {'$exists': 1}}, {"$inc": {"dl_version": 1}})

    def delayed_requests_get(self, url, _params):
        time.sleep(0.3)
        return requests.get(url, params=_params)

    def shopstyle_fields_update(self, prod):
        """
        this func. updates only shopstyle's fields of a product in the DB
        :param prod: dictionary of a DB product
        :return: Nothing, void function
        """
        ss_fields_dict = {}
        for field in prod:
            ss_fields_dict[field] = prod[field]
        self.db.shoes.update({'id': prod['id']}, {'$set': ss_fields_dict})

    def insert_and_fingerprint(self, prod):
        """
        this func. inserts a new product to our DB and runs TG fingerprint on it
        :param prod: dictionary of shopstyle product
        :return: Nothing, void function
        """
        db_fingerprint_nadav.run_fp_on_db_product(prod)  # TODO: understand witch function does that
        self.db.shoes.insert(prod)  # maybe try-except

    def db_update(self, prod):
        current_dl_version = self.db.globals.find_one({'dl_version': {'$exists': 1}})['dl_version']
        # case 1: new product - try to update, if does not exists, insert a new product and add our fields
        prod_in_products = self.db.shoes.find_one({"id": prod["id"]})
        if prod_in_products is None:
            # case 1.1: try finding this product in the archive
            prod_in_archive = self.db.archive.find_one({'id': prod["id"]})
            if prod_in_archive is None:
                self.insert_and_fingerprint(prod)
            else:  # means the item is in the archive
                if prod_in_archive["fp_version"] is constants.fingerprint_version:
                    self.db.shoes.insert(prod_in_archive)
                    self.shopstyle_fields_update(prod)
                else:
                    self.insert_and_fingerprint(prod)
                self.db.archive.remove({'id': prod["id"]})
        else:
            # case 2: the product was found in our db, and maybe should be modified
            # Thus - update only shopstyle's fields
            if prod_in_products["lastModified"] != prod["lastModified"]:  # assuming shopstyle update it correctly
                self.shopstyle_fields_update(prod)
        self.db.shoes.update({'id': prod["id"]}, {'$set': {'dl_version': current_dl_version}})


class UrlParams(collections.MutableMapping):
    """This is current designed specifically for the ShopStyle API where there can be multiple "fl" parameters,
    although it could easily be made more modular, where a list of keys which can have multiple values is predefined,
    or the option (bool multivalue) is supplied on new entry creation...
    """

    def __init__(self, params_dict=dict(), _fl_list=list()):
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
