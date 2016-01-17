__author__ = 'yonatan'

from . import constants

db = constants.db


def getItems(idx):
    items = db.images.find().batch_size(100)
    start = int(idx)
    batch = []
    for i in range(start, 100 + start):
        tmp_item = items[i]
        tmp = {"item_urls": tmp_item["image_urls"],
               "people": tmp_item["people"]}
        for x, candidate in enumerate(tmp["people"]):
            for y, cand_item in enumerate(candidate["items"]):
                tmp["people"][x]["items"][y]["category"] = cand_item['category']
                tmp["people"][x]["items"][y]["saved_date"] = cand_item['saved_date']
                for w in range(10):
                    tmp["people"][x]["items"][y]["top10"][w] = \
                    cand_item['similar_results'][w]["image"]["sizes"]["XLarge"]["url"]
        batch.append(tmp)
    return batch
