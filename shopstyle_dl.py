__author__ = 'yonatan'

import time
import datetime
import sys

import requests
from rq import Queue

from DBDworker import delayed_requests_get
from DBDworker import download_products
import constants

q = Queue('DBD', connection=constants.redis_conn)

BASE_URL = "http://api.shopstyle.com/api/v2/"
BASE_URL_PRODUCTS = BASE_URL + "products/"
PID = "uid900-25284470-95"
TG_FIELDS = ["fp_version", "fingerprint", "fp_date", "human_bb", "bounding_box"]
FILTERS = ["Brand", "Retailer", "Price", "Color", "Size", "Discount"]
MAX_RESULTS_PER_PAGE = 50
MAX_OFFSET = 5000
MAX_SET_SIZE = MAX_OFFSET + MAX_RESULTS_PER_PAGE
db = constants.db
relevant = constants.db_relevant_items
fp_version = constants.fingerprint_version


class ShopStyleDownloader:
    def __init__(self):
        self.db = db
        self.current_dl_date = str(datetime.datetime.date(datetime.datetime.now()))

    def db_download(self, collection):
        if self.db.download_data.find({"criteria": collection}).count() > 0:
            self.db.download_data.delete_one({"criteria": collection})
        self.db.download_data.insert_one({"criteria": collection,
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
                                          "total_dl_time(min)": "still in process",
                                          "last_request": time.time()})
        self.db.drop_collection("fp_in_process")
        self.db.fp_in_process.insert_one({})
        self.db.fp_in_process.create_index("id")
        self.db.dl_cache.delete_many({})
        self.db.dl_cache.create_index("filter_params")
        root_category, ancestors = self.build_category_tree(collection)
        cats_to_dl = [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            self.download_category(cat, collection)
        self.wait_for(collection)
        self.db.download_data.find_one_and_update({"criteria": collection},
                                                  {'$set': {"end_time": datetime.datetime.now()}})
        tmp = self.db.download_data.find({"criteria": collection})[0]
        total_time = abs(tmp["end_time"] - tmp["start_time"]).total_seconds()
        self.db.download_data.find_one_and_update({"criteria": collection},
                                                  {'$set': {"total_dl_time(min)": str(total_time / 60)[:5]}})
        del_items = self.db[collection].delete_many({'fingerprint': {"$exists": False}})
        # print str(del_items.deleted_count) + ' items without fingerprint were deleted!\n'
        self.db.drop_collection("fp_in_process")
        # print collection + " " + type + " DOWNLOAD DONE!!!!!\n"

    def wait_for(self, collection):
        # print "Waiting for 45 min before first check"
        total_items_before = self.db[collection].count()
        time.sleep(2700)
        total_items_after = self.db[collection].count()
        check = 0
        # checking if there is still change in the total items count
        while total_items_before != total_items_after:
            if check > 36:
                break
            # print "\ncheck number " + str(check)
            # print "\nfp workers didn't finish yet\nWaiting 5 min before checking again\n"
            check += 1
            time.sleep(300)
            total_items_before = total_items_after
            total_items_after = self.db[collection].count()

    def build_category_tree(self, collection):
        parameters = {"pid": PID, "filters": "Category"}
        if collection == "products_jp":
            parameters["site"] = "www.shopstyle.co.jp"

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
            parameters["cat"] = anc["id"]
            response = delayed_requests_get(BASE_URL_PRODUCTS + "histogram", parameters, collection)
            hist = response.json()["categoryHistogram"]
            # save count for each category
            for cat in hist:
                self.db.categories.update({"id": cat["id"]}, {"$set": {"count": cat["count"]}})
        return root_category, ancestors

    def download_category(self, category_id, collection):
        if category_id not in relevant:
            return

        category = self.db.categories.find_one({"id": category_id})
        if "count" in category and category["count"] <= MAX_SET_SIZE:
            # print("Attempting to download: {0} products".format(category["count"]))
            # print("Category: " + category_id)
            q.enqueue(download_products, filter_params={"pid": PID, "cat": category_id}, coll=collection)

        elif "childrenIds" in category.keys():
            # print("Splitting {0} products".format(category["count"]))
            # print("Category: " + category_id)
            for child_id in category["childrenIds"]:
                self.download_category(child_id, collection)
        else:
            initial_filter_params = {"pid": PID, "cat": category[
                "id"]}  # UrlParams(params_dict={"pid": PID, "cat": category["id"]})
            self.divide_and_conquer(initial_filter_params, 0, collection)
            # self.archive_products(category_id)  # need to count how many where sent to archive

    def divide_and_conquer(self, filter_params, filter_index, collection):
        """Keep branching until we find disjoint subsets which have less then MAX_SET_SIZE items"""
        if filter_index >= len(FILTERS):
            # TODO: determine appropriate behavior in this case
            # print "ran out of FILTERS"
            return
        filter_params["filters"] = FILTERS[filter_index]

        histogram_response = delayed_requests_get(BASE_URL_PRODUCTS + "histogram", filter_params, collection)
        # TODO: check the request worked here
        try:
            histogram_results = histogram_response.json()
            # for some reason Category doesn't come with a prefix...

            filter_prefix = histogram_results["metadata"].get("histograms")[0].get("prefix")

            hist_key = FILTERS[filter_index].lower() + "Histogram"
            if hist_key not in histogram_results:
                raise RuntimeError("No hist_key {0} in histogram: \n{1}".format(hist_key, histogram_results))
        except:
            # print "Could not get histogram for filter_params: {0}".format(filter_params.encoded())
            # print "Attempting to downloading first 5000"
            # TODO: Better solution for this
            q.enqueue(download_products, filter_params=filter_params, coll=collection)
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
                    q.enqueue(download_products, filter_params=subset_filter_params, coll=collection)
                else:
                    # print "Splitting: {0} products".format(subset["count"])
                    # print "Params: {0}".format(subset_filter_params.encoded())
                    self.divide_and_conquer(subset_filter_params, filter_index + 1, collection)


if __name__ == "__main__":

    col = "products"
    if len(sys.argv) == 2:
        col = col + sys.argv[1]
    print ("@@@ Shopstyle Download @@@\n you choose to update the " + col + " collection")
    update_db = ShopStyleDownloader()
    update_db.db_download(col)

    print "Daily Update Finished!!!"
