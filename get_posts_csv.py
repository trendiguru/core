__author__ = 'liorsabag'

from bson import json_util
import csv
from .constants import db


def find_image_url_by_id(item_id, search_results):
    if isinstance(item_id, basestring) and item_id.isdigit():
        item_id = int(item_id)
    for search_result in search_results:
        if search_result["id"] == item_id:
            return search_result["imageURL"]


def main():
    all_posts_cursor = db.posts.find()
    flattened_posts = []
    for post in all_posts_cursor:
        post = json_util.loads(json_util.dumps(post))
        r_dict = {}
        for item in post["items"]:
            if "topResults" in item:
                r_dict["imageURL"] = post["imageURL"]
                r_dict["boundingBox"] = item["boundingBox"]
                r_dict["categoryId"] = item["categoryId"]

                result_num = 1
                for result in item["topResults"]:
                    r_dict["similar" + str(result_num)] = find_image_url_by_id(result, item["searchResults"])
                    result_num += 1

                result_num = 1
                for result in item["bottomResults"]:
                    r_dict["other" + str(result_num)] = find_image_url_by_id(result, item["searchResults"])
                    result_num += 1

                flattened_posts.append(r_dict)
                print "appended " + r_dict["imageURL"]

    headers = ["imageURL", "boundingBox", "categoryId"]
    similars = ["similar" + str(i) for i in range(1, 6)]
    others = ["other" + str(i) for i in range(1, 201)]
    headers = headers + similars + others

    with open('flattened_posts.csv', 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, headers)
        w.writeheader()
        for post in flattened_posts:
            w.writerow(post)


if __name__ == "__main__":
    main()