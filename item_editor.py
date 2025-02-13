__author__ = 'liorsabag'

from bson.objectid import ObjectId
import fingerprint_core
import Utils
import logging
from .constants import db

BB_ALLOWANCE = 0.05


def find(image_url, number_of_items=None):
    """
    Find a post in the db given an image url.
    Searches first by url, then by fingerprint (if not found)

    :param image_url: string, url of the image
    :param number_of_items: max number of items to return (I think this is a mistake)
    :return: a dictionary containing all post info
    """
    post = db.posts.find_one({"imageURL": image_url})
    if not post:
        fingerprint = fingerprint_core.fp(Utils.get_cv2_img_array(image_url)).tolist()
        post = db.posts.find_one({"fingerprint": fingerprint})
    # TODO: check this if - on second look, doesn't really make sense
    # (why trim num of items and not number of results)
    if post and "items" in post and number_of_items is not None:
        post["items"] = post["items"][0:number_of_items]
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
    items = post.get("items", [])

    # either create a new bb or change an existing one
    # check if its an existing one

    # query_bb = post.get("boundingBox", [])
    # bb_index = find_item_by_bb(query_bb, items)
    item_index = find_item_index(item_data, items)

    # if an item with this bb exists, replace it...
    if item_index != -1:
        del items[item_index]
    # delete image_url to prevent data duplication between doc and subdoc
    del item_data["imageURL"]
    # TODO: rethink deleting id always
    item_data["id"] = len(items)
    items.append(item_data)

    # add updated items array to post
    db.posts.update({"_id": post["_id"]}, {"$set": {"items": items}})


def find_item_index(item_data, items):
    item_index = -1
    if "id" in item_data:
        for i in range(0, len(items)):
            if items[i]["id"] == item_data["id"]:
                item_index = i
                break
    else:
        item_index = find_item_by_bb(item_data.get("boundingBox", []), items)
    return item_index


def find_item_by_bb(query_bb, items):
    index = -1
    if query_bb:
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
    post = db.posts.find_one({"imageURL": url})
    if post is None:
        fingerprint = fingerprint_core.fp(Utils.get_cv2_img_array(url)).tolist()
        post = db.posts.find_one({"fingerprint": fingerprint})
        if post is None:
            result = db.posts.insert({"imageURL": url, "fingerprint": fingerprint})
            if type(result) is ObjectId:
                post = db.posts.find_one({"_id": result})
    return post or {}