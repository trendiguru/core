__author__ = 'yonatan'

"""
DB downloader
"""

import time
import datetime

import requests
from rq import Queue

from DBDworker import download_category
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
        self.db.DBD.insert_one({"criteria": "main",
                                "last_request": self.last_request_time,
                                "items_downloaded": 0})
        self.do_fingerprint = do_fingerprint  # check if relevent
        root_category, ancestors = self.build_category_tree()
        self.db.DBD.delete_many({})
        cats_to_dl = [anc["id"] for anc in ancestors]
        for cat in cats_to_dl:
            q.enqueue(download_category, category_id=cat)
            # self.download_category(cat)

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

    def delayed_requests_get(self, url, _params):
        sleep_time = max(0, 0.1 - (time.time() - self.db.DBD.find()[0]["last_request_time"]))
        time.sleep(sleep_time)
        self.db.DBD.find_one_and_update({"criteria": "main"},
                                        {'$set': {"last_request": time.time()}})
        return requests.get(url, params=_params)


if __name__ == "__main__":
    update_db = ShopStyleDownloader()
    update_db.run_by_category()
