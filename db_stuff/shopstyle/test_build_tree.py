from time import time, sleep
import requests
from ...constants import db
BASE_URL = "http://api.shopstyle.com/api/v2/"
PID = "uid900-25284470-95"
categories = db.categories
last_request_time = time()


def old_method():
    strt = time()
    parameters = {"pid": PID, "filters": "Category"}

    # download all categories
    category_list_response = requests.get(BASE_URL + "categories", params=parameters)
    category_list_response_json = category_list_response.json()
    root_category = category_list_response_json["metadata"]["root"]["id"]
    category_list = category_list_response_json["categories"]
    categories.remove({})
    categories.insert(category_list)
    # find all the children
    for cat in categories.find():
        categories.update_one({"id": cat["parentId"]}, {"$addToSet": {"childrenIds": cat["id"]}})
    # get list of all categories under root - "ancestors"
    ancestors = []
    for c in categories.find({"parentId": root_category}):
        ancestors.append(c)
    # let's get some numbers in there - get a histogram for each ancestor
    for anc in ancestors:
        parameters["cat"] = anc["id"]
        response = delayed_requests_get(BASE_URL + "products/histogram", parameters)
        hist = response.json()["categoryHistogram"]
        # save count for each category
        for cat in hist:
            categories.update_one({"id": cat["id"]}, {"$set": {"count": cat["count"]}})
    end_time = time()
    print (end_time - strt)
    return root_category, ancestors


def new_method():
    strt = time()
    parameters = {"pid": PID, "filters": "Category"}

    # download all categories
    category_list_response = requests.get(BASE_URL + "categories", params=parameters)
    category_list_response_json = category_list_response.json()
    root_category = category_list_response_json["metadata"]["root"]["id"]
    category_list = category_list_response_json["categories"]
    # find all the children
    # for cat in self.categories.find():
    #     self.categories.update_one({"id": cat["parentId"]}, {"$addToSet": {"childrenIds": cat["id"]}})
    # this was replaced because it makes many calls to the db
    category_ids = []
    parent_ids = []
    ancestors = []
    for cat in category_list:
        category_ids.append(cat['id'])
        parent_ids.append(cat['parentId'])
        cat['childrenIds'] = []
    for child_idx, parent in enumerate(parent_ids):
        if parent == root_category:
            ancestors.append(category_list[child_idx])
        if category_ids.__contains__(parent):
            parent_idx = category_ids.index(parent)
            category_list[parent_idx]['childrenIds'].append(category_list[child_idx]['id'])

    # let's get some numbers in there - get a histogram for each ancestor
    for anc in ancestors:
        parameters["cat"] = anc["id"]
        response = delayed_requests_get('{}products/histogram'.format(BASE_URL), parameters)
        hist = response.json()["categoryHistogram"]
        # save count for each category
        for cat in hist:
            cat_idx = category_ids.index(cat['id'])
            category_list[cat_idx]['count'] = cat['count']

    categories.remove({})
    categories.insert(category_list)
    end_time = time()
    print (end_time-strt)
    return root_category, ancestors


def delayed_requests_get(url, params):
    global last_request_time
    sleep_time = max(0, 0.1 - (time() - last_request_time))
    sleep(sleep_time)
    last_request_time = time()
    return requests.get(url, params=params)