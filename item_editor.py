__author__ = 'liorsabag'

from bson.objectid import ObjectId
import fingerprint_core
import Utils
import pymongo
db = pymongo.MongoClient().mydb
BB_ALLOWANCE = 0.05


def find(image_url):
    post = db.posts.find_one({"imageURL": image_url})
    if post == {}:
        fingerprint = fingerprint_core.fp(Utils.get_cv2_img_array(image_url))
        post = db.posts.find_one({"fingerprint": fingerprint})
    return post


def save(item_data):
    """
    Each doc in the db will be a post (image).
    There will be an array of sub-documents, "items", where each item is identified by its bb (relative to image).
    Each item will have top results saved by url/id

    Are we doing a smart or dumb server?
    Should the server handle resizing images, calculating relative positions etc..?
    For now, assume calculations are client side

    Given new post data, we have to retrieve the relevant image using URL or fingerprint,
    then check if we're appending/modifying an existing item or creating a new one based on bb.

    :param item_data: dictionary of data from the Match Editor
    :return: success...
    """
    # get the post
    post = find_or_create_post(item_data["imageURL"])

    # get items
    items = []
    if "items" in post:
        items = post["items"]

    # either create a new bb or change an existing one
    # check if its an existing one
    query_bb = post["boundingBox"]
    bb_index = find_item_by_bb(query_bb, items)
    # if an item with this bb exists, replace it...
    if bb_index != -1:
        del items[bb_index]
    # delete image_url to prevent data duplication between doc and subdoc
    del item_data["imageURL"]
    item_data["id"] = len(items)
    items.append(item_data)

    # add updated items array to post
    db.posts.update({"_id": post["_id"]}, {"item": items})


def find_item_by_bb(query_bb, items):
    index = -1
    for i in range(0, len(items)):
        bb = items[i]["bb"]
        if [is_within_allowance_of(query_bb[j], bb[j], BB_ALLOWANCE) for j in range(0, len(bb))].all():
            # More or less same bb
            index = i
            break
    return index


def is_within_allowance_of(query, target, allowance):
    return ((1 - allowance) * target) < query < ((1 + allowance) * target)


def find_or_create_post(url):
    """
    Search for the post by image - first by url, and if not by  by fp.
    :param url: image url - this is coming directly from the web interface so it's all we'll ever get.
    :return: post
    """
    post = db.posts.find_one({"image_url": url})
    if post == {}:
        fingerprint = fingerprint_core.fp(Utils.get_cv2_img_array(url))
        post = db.posts.find_one({"fingerprint": fingerprint})
        if post == {}:
            result = db.posts.insert({"image_url": url, "fingerprint": fingerprint})
            if type(result) is ObjectId:
                post = db.posts.find_one({"_id": result})
    return post