__author__ = 'yonatan'

import datetime

import bson

from . import constants

db = constants.db


def getItems(last_id, date_filter):
    filters = {}
    if date_filter != "":
        filters["saved_date"] = {"$gt": datetime.datetime.strptime(date_filter, "%Y-%m-%d")}
    if last_id == '0':
        items = db.images.find(filters).limit(100)
    else:
        filters["_id"] = {"$gt": bson.ObjectId(last_id)}
        items = db.images.find(filters).limit(100)
    batch = []
    max_range = min(100, items.count())
    for i in range(0, max_range):
        tmp_item = items[i]
        tmp = {"item_urls": tmp_item["image_urls"],
               "saved_date": tmp_item["saved_date"]}
        people = []
        for x, candidate in enumerate(tmp_item["people"]):
            items4people = []
            try:
                for y, cand_item in enumerate(candidate["items"]):
                    itemCategory = cand_item['category']
                    top10 = []
                    try:
                        for w in range(10):
                            top10.append(cand_item['similar_results'][w]["image"]["sizes"]["XLarge"]["url"])
                    except:
                        try:
                            for w in range(10):
                                top10.append(cand_item['similar_results'][w]["images"]["XLarge"])
                        except:
                            pass
                    dict = {'category': itemCategory,
                            'top10': top10}
                    items4people.append(dict)
                people.append(items4people)
            except:
                pass
        if len(people) == 0:
            continue
        tmp["people"] = people
        batch.append(tmp)
    last_id = tmp_item["_id"]
    tmp = {"last_id": last_id}
    batch.append(tmp)
    return batch
