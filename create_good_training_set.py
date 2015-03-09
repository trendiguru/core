__author__ = 'liorsabag'

import pymongo
import sys

db = pymongo.MongoClient().mydb


def main(min_num_images, collection_name):
    for doc in db.training.find():
        if len(doc["images"]) >= min_num_images:
            new_doc = {}
            new_doc["_id"] = doc["_id"]
            new_doc["images"] = []
            for i in range(0, min_num_images):
                new_doc["images"].append(doc["images"][i])
            db[collection_name].insert(new_doc)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])