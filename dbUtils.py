# yo yo

from __future__ import print_function

__author__ = 'jeremy'

# builtin
import logging
import rq
import cv2
from bson import objectid, ObjectId
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import time
import tldextract
# ours
import constants
import page_results
from .find_similar_mongo import get_all_subcategories, find_top_n_results
from . import background_removal
from . import Utils
from .constants import db, redis_conn
from falcon import sleeve_client
rq.push_connection(redis_conn)
min_images_per_doc = constants.min_images_per_doc
max_image_val = constants.max_image_val

hash_q = rq.Queue("hash_q")
add_feature = rq.Queue("add_feature")

def lookfor_next_bounded_in_db(current_item=0, current_image=0, only_get_boxed_images=True):
    """
    find next bounded image in db
    :rtype : dictionary
    :input: i, the index of the current item
    :return:url,bb, and skip_it for next unbounded image
    """
    current_item = int(current_item)  # this is needed since js may be returning strings
    current_image = int(current_image)
    print('entered lookfornext_in_db:current item:' + str(current_item) + ' cur img:' + str(
        current_image) + ' only get boxed:' + str(only_get_boxed_images))
    logging.warning('w_entered lookfornext_in_db:current item:' + str(current_item) + ' cur img:' + str(
        current_image) + ' only get boxed:' + str(only_get_boxed_images))

    # training docs contains lots of different images (URLs) of the same clothing item
    training_collection_cursor = db.training.find()  # .sort _id
    # doc = next(training_collection_cursor, None)
    i = current_item
    doc = training_collection_cursor[i]
    if doc is None:
        i = 0
    doc = training_collection_cursor[0]
    if doc is None:
        logging.warning('couldnt get any doc from db')
        return None
    while doc is not None:
        # print('doc:' + str(doc))
        # logging.warning('calling lookfor_next_bounded, index='+str(i)+' image='+str(current_image))
        answers = lookfor_next_bounded_image(doc, image_index=current_image,
                                             only_get_boxed_images=only_get_boxed_images)
        # logging.warning('returned from  lookfor_next_bounded')
        if answers is not None:
            answers['id'] = str(doc['_id'])
            answers['item_index'] = i
            if only_get_boxed_images:
                try:
                    if answers["bb"] is not None:  # got a good bb
                        logging.debug('exiting lookfornext 1, answers:' + str(answers))
                        return answers
                except KeyError, e:
                    print('keyerror on key "%s" which probably does not exist' % str(e))
                    # go to next doc since no bb was found in this one
            else:
                logging.debug('exiting lookfornext 2, answers:' + str(answers))
                return answers
        i = i + 1
        current_image = 0
        doc = training_collection_cursor[i]
        logging.warning("no bounded image found in current doc, trying next")
    return {'error': 0, 'message': "No bounded bb found in db"}


def lookfor_next_image_in_regular_db(current_item, only_get_boxed_images=False,
                                     only_get_unboxed_images=True, skip_if_marked_to_skip=True):
    """
    find next image in db
    :rtype : dictionary
    :input: i, the index of the current item, only_get_box, only_get_unboxed
    :return:url,bb, and skip_it for next unbounded image
    """
    current_item = int(current_item)  # this is needed since js may be returning strings
    print(
    'entered lookfornext_in_db:current item:' + str(current_item) + ' only get boxed:' + str(only_get_boxed_images))
    logging.warning('w_entered lookfornext_in_db:current item:' + str(current_item) + ' only get boxed:' + str(
        only_get_boxed_images))

    # training docs contains lots of different images (URLs) of the same clothing item
    training_collection_cursor = db.training.find()  # .sort _id
    # doc = next(training_collection_cursor, None)
    i = current_item
    doc = training_collection_cursor[i]
    if doc is None:
        i = 0
    doc = training_collection_cursor[0]
    if doc is None:
        logging.warning('couldnt get any doc from db')
        return None
    while doc is not None:
        # print('doc:' + str(doc))
        # logging.warning('calling lookfor_next_bounded, index='+str(i)+' image='+str(current_image))
        answers = lookfor_next_bounded_image(doc, image_index=current_item,
                                             only_get_boxed_images=only_get_boxed_images)
        # logging.warning('returned from  lookfor_next_bounded')
        if answers is not None:
            answers['id'] = str(doc['_id'])
            answers['item_index'] = i
            if only_get_boxed_images:
                try:
                    if answers["bb"] is not None:  # got a good bb
                        logging.debug('exiting lookfornext 1, answers:' + str(answers))
                        return answers
                except KeyError, e:
                    print('keyerror on key "%s" which probably does not exist' % str(e))
                    # go to next doc since no bb was found in this one
            else:
                logging.debug('exiting lookfornext 2, answers:' + str(answers))
                return answers
        i = i + 1
        current_image = 0
        doc = training_collection_cursor[i]
        logging.warning("no bounded image found in current doc, trying next")
    return {'error': 0, 'message': "No bounded bb found in db"}


def lookfor_next_unbounded_image(queryobject):
    n = 0
    got_unbounded_image = False
    urlN = None  # if nothing eventually is found None is returned for url
    images = queryobject["images"]
    # print('utils.py:images:'+str(images))
    logging.debug('Utils.py(debug):images:' + str(images))

    if len(images) < min_images_per_doc:  # don't use docs with too few images
        return (None)
    print('# images:' + str(len(images)))
    for entry in images:
        if 'skip_image' in entry:
            if entry['skip_image'] == True:
                print('utils.py:image is marked to be skipped')
                logging.debug('Utils.py(debug):image is marked to be skipped')
                continue
            else:
                print('utils.py:image is NOT marked to be skipped')
                logging.debug('Utils.py(debug):image is NOT marked to be skipped')
        if not 'human_bb' in entry:  # got a pic without a bb
            urlN = entry['url']
            got_unbounded_image = True
            print('utils.py:no human bb entry for:' + str(entry))
            return (urlN)
        elif entry["human_bb"] is None:
            urlN = entry['url']
            got_unbounded_image = True
            print('utils.py:human_bb is None for:' + str(entry))
            return (urlN)
        elif not isinstance(entry["human_bb"], list):
            urlN = entry['url']
            got_unbounded_image = True
            print('utils.py:illegal bb!! (not a list) for:' + str(entry))
            return (urlN)
        elif not (Utils.legal_bounding_box(entry["human_bb"])):
            urlN = entry['url']
            got_unbounded_image = True
            print('utils.py:bb is not legal (too small) for:' + str(entry))
            return (urlN)
        else:
            urlN = None
            got_unbounded_image = False
            print('utils.py:image is bounded :(')
            logging.debug('image is bounded.....')
    return (urlN)


def lookfor_next_bounded_image(queryobject, image_index=0, only_get_boxed_images=True):
    """
    finds next image that has bounding box
    :param queryobject: this is a db entry
    :return:url, skip (whether or not to skip)
    """
    logging.warning(
        'Utils.lookfor_next_bounded, image_index:' + str(image_index) + ' only_boxed:' + str(only_get_boxed_images))

    answers = {}
    n = 0
    skip_image = False
    got_unbounded_image = False
    urlN = None  # if nothing eventually is found None is returned for url
    if not 'images' in queryobject:
        logging.debug('Utils.py(debug):no images in input:' + str(queryobject))
        return None
    images = queryobject["images"]
    if images is None:
        logging.debug('Utils.py(debug):images is None!!')
        return None
    # print('utils.py:images:'+str(images))
    logging.debug('Utils.py(debug):images:' + str(images))
    # check for suitable number of images in doc - removed since i wanna check all the bbs
    # if len(images) < min_images_per_doc:  # don't use docs with too few images
    # print('# images is too small:' + str(len(images)) + ' found and ' + str(min_images_per_doc) + ' are required')
    # logging.debug('Utils.py(debug):image is marked to be skipped')
    # return None
    print(
        '# images:' + str(len(images)) + ' image_index:' + str(image_index) + ' only boxed:' + str(
            only_get_boxed_images))
    try:
        answers["_id"] = str(queryobject["_id"])
    except KeyError, e:
        print('keyerror on key "%s" which probably does not exist' % str(e))
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    try:
        answers["product_title"] = queryobject["Product Title"]
    except KeyError, e:
        print('keyerror on key "%s" which probably does not exist' % str(e))
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    try:
        answers["product_url"] = queryobject["Product URL"]
    except KeyError, e:
        print('keyerror on key "%s" which probably does not exist' % str(e))
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    answers['skip_image'] = False
    if image_index == max_image_val:  # max_image_val is a code meaning get the last image
        image_index = len(images) - 1
    if image_index >= len(images):  # index starts at 0
        #image_index=0
        print('utils - past index, returning None')
        return None  #get the next item if we're past last image
    for i in range(image_index, len(images)):
        entry = images[i]
        answers['image_index'] = i
        answers['url'] = entry['url']
        if 'skip_image' in entry:
            answers['skip_image'] = entry['skip_image']

        if 'human_bb' in entry and entry['human_bb'] is not None:  # got a pic with a bb
            answers['bb'] = entry['human_bb']
            # answers['x'] = entry['human_bb'][0]
            #            answers['y'] = entry['human_bb'][1]
            #            answers['w'] = entry['human_bb'][2]
            #            answers['h'] = entry['human_bb'][3]
            return answers

        elif only_get_boxed_images == False:  # no human_bb in this entry but its ok, return anyway
            return answers

    print('utils.lookfor_next_bounded_image:no bounded image found in this doc:(')
    logging.debug('utils.lookfor_next_bounded_image - no bounded image found in this doc')
    return None


def insert_bb_into_training_db(receivedData):
    bb = receivedData['bb']
    image_url = receivedData["url"]
    if 'skip_image' in receivedData:
        skip_image = receivedData["skip_image"]
    else:
        skip_image = False
    current_image = receivedData["current_image"]
    current_item = receivedData["current_item"]
    id = receivedData["id"]

    if current_item is None or id is None:
        return {"success": 0, "error": "wasnt given an id or current_iutem to work with"}

    # id = vars['_id']
    print(
        'default.py:bb:' + str(bb) + ' imageurl:' + str(image_url) + ' skip:' + str(
            skip_image) + ' current image:' + str(
            current_image) + ' current item:' + str(current_item))
    logging.debug(
        'bb:' + str(bb) + ' imageurl:' + str(image_url) + ' skip:' + str(skip_image) + ' current image:' + str(
            current_image) + ' current item:' + str(current_item))
    image_dict = {}
    result_dict = {}
    # find the document  - can do this either by id like thie

    if db is None:
        return {"success": 0, "error": "could not get db"}
    trainingdb = db.training
    if trainingdb is None:
        return {"success": 0, "error": "could not get trainingdb"}
    doc = trainingdb.find_one({'_id': objectid.ObjectId(id)})
    # or can do it by looking for item # lke this
    # training_collection_cursor = db.training.find()   #.sort _id
    # doc = training_collection_cursor[current_item]
    if not doc:
        return {"success": 0, "error": "could not get doc with specified id" + str(id)}
    i = 0
    for image in doc["images"]:
        if image["url"] == image_url:
            image["human_bb"] = bb
            image["skip_image"] = skip_image
            print('default.py:new image:' + str(image))
            # subtle error - if two images have same url, only one will get updated causing the other to get shown forever after
            # therefore dont break here but rather continue adding bb for each image having same url
            # save edited doc
            # TODO: check error on updating
            write_result = db.training.update({"_id": objectid.ObjectId(id)}, {"$set": {"images": doc['images']}})
            if current_image != i:
                print('inconsistency - item number ' + str(i) + '+doesnt match')
                logging.warning('inconsistency - item number ' + str(i) + ' doesnt match')
            print('write result:' + str(write_result))
            return {"success": 1}
        i = i + 1
    return {"success": 0, "error": "could not find image w. url:" + str(image_url) + " in current doc:" + str(doc)}

# db = pymongo.MongoClient().mydb
# if db is None:
#        return {"success": 0, "error": "could not get db"}
#    trainingdb = db.training
#    if trainingdb is None:
#        return {"success": 0, "error": "could not get trainingdb"}


def insert_feature_bb_into_db(new_feature_bb, itemID=id, db=None):
    if db is None:
        return {"success": 0, "error": "could not get db"}
    trainingdb = db.training
    if trainingdb is None:
        return {"success": 0, "error": "could not get trainingdb"}

    doc = trainingdb.find_one({'_id': objectid.ObjectId(id)})

    if not doc:
        return {"success": 0, "error": "could not get doc with specified id" + str(id)}
    i = 0
    write_result = db.training.update({"_id": objectid.ObjectId(id)}, {"$set": {"images": doc['images']}})
    return {"success": 1, 'message': "ok"}


def fix_all_bbs_in_db(use_visual_output=True):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        return {"success": 0, "error": "could not get db"}
    training_collection_cursor = db.training.find()
    print('returned cursor')
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    j = 0
    while doc is not None:
        print('doc:' + str(doc))
        images = doc['images']
        print('checking doc #' + str(j + 1))
        i = 0
        for image in images:
            image_url = image["url"]
            if 'skip_image' in image:
                if image['skip_image'] == True:
                    print('marked for skip:' + str(i), end='\r')
                    continue
            img_arr = Utils.get_cv2_img_array(image_url, convert_url_to_local_filename=True, download=True,
                                              download_directory='images')
            if not Utils.is_valid_image(img_arr):
                print('img is not valid (=None or too small')
                continue

            if 'human_bb' in image:
                i = i + 1
                height, width = img_arr.shape[0:2]
                bb = image["human_bb"]
                if bb is None:
                    print('bb is None')
                    continue

                cv2.rectangle(img_arr, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=[0, 0, 255],
                              thickness=2)
                cv2.imshow('img', img_arr)
                k = cv2.waitKey(50) & 0xFF
                if not Utils.bounding_box_inside_image(img_arr, bb):
                    print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
                        image) + ' h,w:' + str(height) + ',' + str(width))
                    print('h,w:' + str(height) + ',' + str(width))
                    bb[0] = max(0, bb[0])  # if less than zero
                    bb[0] = min(bb[0], width - 1)  # if greater than width
                    bb[2] = max(0, bb[2])  # if less than 0
                    bb[2] = min(bb[2], width - bb[0] - 1)  # the -1 is just to make sure, prob unneeded

                    bb[1] = max(0, bb[1])  # if less than zero
                    bb[1] = min(bb[1], height - 1)  # if greater than height
                    bb[3] = max(0, bb[3])  # if less than zero
                    bb[3] = min(bb[3], height - bb[1] - 1)  # the -1 is just to make sure, prob unneeded
                    print('suggested replacement:' + str(bb))
                    raw_input('got one')
                    image["human_bb"] = bb
                    id = str(doc['_id'])
                    write_result = db.training.update({"_id": objectid.ObjectId(id)},
                                                      {"$set": {"images": doc['images']}})
                    # TODO: check error on updating
                    print('write result:' + str(write_result))
                else:
                    print('got good bb, i=' + str(i), end='\r', sep='')

        j = j + 1
        doc = next(training_collection_cursor, None)

    return {"success": 1}


def show_all_bbs_in_db(use_visual_output=True):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        return {"success": 0, "error": "could not get db"}
    training_collection_cursor = db.training.find()
    print('returned cursor')
    assert (training_collection_cursor)  # make sure training collection exists
    doc = next(training_collection_cursor, None)
    i = 0
    j = 0
    while doc is not None:
        print('doc:' + str(doc))
        images = doc['images']
        print('checking doc #' + str(j + 1))
        print(doc)
        i = 0
        for image in images:
            image_url = image["url"]
            if 'skip_image' in image:
                if image['skip_image'] == True:
                    print('marked for skip:' + str(i), end='\r')
                    continue
            img_arr = Utils.get_cv2_img_array(image_url, convert_url_to_local_filename=True, download=True,
                                              download_directory='images')
            if not Utils.is_valid_image(img_arr):
                print('img is not valid (=None or too small')
                continue

            if 'human_bb' in image:
                i = i + 1
                height, width = img_arr.shape[0:2]
                bb = image["human_bb"]
                if bb is None:
                    print('bb is None')
                    continue

                if not Utils.bounding_box_inside_image(img_arr, bb):
                    print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
                        image) + ' h,w:' + str(height) + ',' + str(width))

                    if use_visual_output:
                        # cv2.rectangle(img_arr, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=[0,255,0], thickness=2)
                        cv2.imshow('im1', img_arr)
                        k = cv2.waitKey(0) & 0xFF
                else:
                    print('got good bb, i=' + str(i), end='\r', sep='')

                    if use_visual_output:
                        cv2.rectangle(img_arr, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=[0, 255, 0],
                                      thickness=2)
                        cv2.imshow('im1', img_arr)
                        k = cv2.waitKey(0) & 0xFF
                        # raw_input('waiting for input')
        j = j + 1
        doc = next(training_collection_cursor, None)

    return {"success": 1}


def step_thru_db(use_visual_output=True, collection='products'):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    dbstring = 'db.' + collection
    # cursor = dbstring.find()
    # cursor = db.training.find()
    # look in defaults.py  how this is done
    cursor = db.products.find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get colelction"}
    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        print('doc:' + str(doc))
        for topic in doc:
            try:
                print(str(topic) + ':' + str(doc[topic]))
            except UnicodeEncodeError:
                print('unicode encode error')

        large_url = doc['image']['sizes']['Large']['url']
        print('large img url:' + str(large_url))
        if use_visual_output:
            img_arr = Utils.get_cv2_img_array(large_url)
            if 'bounding_box' in doc:
                if Utils.legal_bounding_box(doc['bounding_box']):
                    bb1 = doc['bounding_box']
                    cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), [255, 255, 0],
                                  thickness=2)
            cv2.imshow('im1', img_arr)
            k = cv2.waitKey(50) & 0xFF
        if 'categories' in doc:
            try:
                print('cats:' + str(doc['categories']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['categories']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        if 'description' in doc:
            try:
                print('desc:' + str(doc['description']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['description']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}


def step_thru_images_db(use_visual_output=True, collection='images'):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.images.find()
    print('returned cursor')
    print('n_items:' + str(cursor.count()))

    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get colelction"}
    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        print('doc:' + str(doc))
        for topic in doc:
            try:
                print(str(topic) + ':' + str(doc[topic]))
            except UnicodeEncodeError:
                print('unicode encode error')
                print(str(topic).encode('utf-8'))

                # large_url = doc['image']['sizes']['Large']['url']
            #        print('large img url:' + str(large_url))
            #        if use_visual_output:
            #            img_arr = Utils.get_cv2_img_array(large_url)
            #            if 'bounding_box' in doc:
            #                if Utils.legal_bounding_box(doc['bounding_box']):
            #                    bb1 = doc['bounding_box']
            #                    cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), [255, 255, 0],
            #                                  thickness=2)
            #            cv2.imshow('im1', img_arr)
            #            k = cv2.waitKey(50) & 0xFF
        i = i + 1
        doc = next(cursor, None)
        print('')
        # raw_input('enter key for next doc')
    print('finished all docs')
    return {"success": 1}


def step_thru_training_db(use_visual_output=False):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.training.find()
    if cursor is None:  # make sure training collection exists
        print('couldnt get training cursor ')
        return {"success": 0, "error": "could not get training collection"}
    print('got cursor')
    print('n_items:' + str(cursor.count()))


    doc = next(cursor, None)
    i = 0
    while doc is not None:
        print('checking doc #' + str(i + 1))
        # print('doc:' + str(doc))
        for topic in doc:
            try:
                print(str(topic) + ':' + str(doc[topic]))
            except UnicodeEncodeError:
                print('unicode encode error')

        if use_visual_output:
            images = doc['images']
            n = len(images)
            i = 0
            fig = plt.figure()
            all_images = np.zeros((100, 1, 3))
            h, w, d = all_images.shape
            print('allimages shape is {0},{1},{2}'.format(h, w, d))
            urllist = []
            for imagedict in images:
                url = imagedict['url']
                img_arr = Utils.get_cv2_img_array(url)
                urllist.append(url)

                if img_arr is not None:
                    # a=fig.add_subplot(1,n,i)
                    h, w, d = img_arr.shape
                    print('shape is {0},{1},{2}'.format(h, w, d))
                    scale = 100.0 / h
                    resized = cv2.resize(img_arr, (int(w * scale), 100))
                    h, w, d = resized.shape
                    print('resized shape is {0},{1},{2}'.format(h, w, d))
                    all_images = np.hstack((all_images, resized))

                    # fig = plt.figure()
                    cv2.imshow('im1', img_arr)
                    cv2.imshow('im1', all_images)
                    k = cv2.waitKey(50) & 0xFF
                else:
                    print('img arr not good')
                if 'bounding_box' in doc:
                    # if Utils.legal_bounding_box(doc['bounding_box']):
                    # bb1 = doc['bounding_box']
                    # cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), [255, 255, 0],
                    # thickness=2)
                    pass
                i = i + 1
            plt.show()
        if 'categories' in doc:
            try:
                print('cats:' + str(doc['categories']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['categories']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        if 'description' in doc:
            try:
                print('desc:' + str(doc['description']))
            except UnicodeEncodeError:
                print('unicode encode error in description')
                s = doc['description']
                print(s.encode('utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
                # print(unicode(s.strip(codecs.BOM_UTF8), 'utf-8'))
        i = i + 1
        doc = next(cursor, None)
        print('')
        raw_input('enter key for next doc')
    return {"success": 1}


def prune_training_db(use_visual_output=False):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''

    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    cursor = db.training.find()
    if cursor is None:  # make sure training collection exists
        print('couldnt get training cursor ')
        return {"success": 0, "error": "could not get training collection"}
    print('got cursor')
    doc = next(cursor, None)
    i = 0
    j = 0
    skus = []
    indices = []
    ids = []
    while doc is not None:
        print('checking doc #' + str(i + 1))
        # print('doc:' + str(doc))
        # for topic in doc:
        # try:
        # print(str(topic) + ':' + str(doc[topic]))
        #         except UnicodeEncodeError:
        #            print('unicode encode error')
        if '_id' in doc:
            ids.append(doc['_id'])
        if 'SKU ID' in doc:
            sku_id = doc['SKU ID']
            j = j + 1
            skus.append(sku_id)
            indices.append(j)
        i = i + 1
        doc = next(cursor, None)
        print('')

    # raw_input('enter key for next doc')

    print('len skus:{0}, len indices:{1}, len ids:{2}', len(skus), len(indices), len(ids))
    # now look for duplicate skus
    n_duplicates = 0
    for i in range(0, len(skus)):
        for j in range(i + 1, len(skus)):
            if skus[i] == skus[j]:
                n_duplicates = n_duplicates + 1
                print('two identical skus for indices {0},{1}'.format(i, j))
                # print('ids={2},{3}'.format(ids[i],ids[j]))
                # subcategory_id_list = get_all_subcategories(db.categories, category_id)

                #curr_cat = category_collection.find_one({"id": c_id})
                ith = db.training.find_one({'_id': ids[i]})
                jth = db.training.find_one({'_id': ids[j]})
                print('first:' + str(ith))
                print('second:' + str(jth))
                db.training.remove({'_id': ids[j]}, justOne=True)
    print(str(n_duplicates) + ' duplicates found')
    return {"success": 1}


# answers = dbUtils.lookfor_next_unbounded_feature_from_db_category(current_item=current_item,skip_if_marked_to_skip=skip_if_marked_to_skip,which_to_show=which_to_show,filter_type=filter_type,category_id=category_id,word_in_description=word_in_description )


def lookfor_next_unbounded_feature_from_db_category(current_item=0, skip_if_marked_to_skip=True,
                                                    which_to_show='showUnboxed', filter_type='byWordInDescription',
                                                    category_id=None, word_in_description=None, db=None):
    # {"id":"v-neck-sweaters"}  coats
    # query_doc = {"categories": {"shortName":"V-Necks"}}
    print('looking, ftype=' + str(filter_type))

    if db is None:
        # print('dbUtils.get_next_unbounded_feature_from_db - problem getting DB')
        logging.warning('dbUtils.get_next_unbounded_feature_from_db - problem getting DB')
        return None
    # make sure theres a known filter type
    if not filter_type in ['byCategoryID', 'byWordInDescription']:
        # print('couldnt figure out filter type:' + str(filter_type))
        logging.warning('couldnt figure out filter type:' + str(filter_type))
        return {'success': 0, 'error': 'couldnt figure out filter type'}

    # make sure theres a known which_to_show
    if not which_to_show in ['showBoxed', 'showUnboxed', 'showAll']:
        # print('couldnt figure out which_to_show:' + str(which_to_show))
        logging.warning('couldnt figure out which to show:' + str(which_to_show))
        return {'success': 0, 'error': 'couldnt figure out which to show' + str(which_to_show)}

    if filter_type == 'byCategoryID':
        print('filtering by ID')
        query = {"categories": {"$elemMatch": {"id": category_id}}}
        fields = {"categories": 1, "image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1,
                  "id": 1, "feature_bbs": 1}
        filter = category_id


    elif filter_type == 'byWordInDescription':
        print('filtering by word in description')
        if word_in_description == None:
            print('no word given to find in description so finding everything')
            logging.warning('no word given to find in description so finding everything')
            word_in_description = ''
        myregex = ".*" + word_in_description + ".*"
        query = {"description": {'$regex': myregex}}
        fields = {"categories": 1, "image": 1, "human_bb": 1, "fp_version": 1, "bounding_box": 1,
                  "id": 1, "description": 1, "feature_bbs": 1}
        filter = word_in_description

    else:
        print('couldnt figure out filter type:' + str(filter_type))
        return {'success': 0, 'error': 'couldnt figure out filter type'}
    num_processes = 100
    product_cursor = db.products.find(query, fields).batch_size(num_processes)
    if product_cursor is None:
        print('got no docs in lookfor_next_unbounded_feature_from_db_category')
        logging.debug('got no docs in lookfor_next_unbounded_feature_from_db_category')
        return None
    ans = get_first_qualifying_record(product_cursor, which_to_show=which_to_show, filter_type=filter_type,
                                      filter=filter, item_number=current_item,
                                      skip_if_marked_to_skip=skip_if_marked_to_skip)

    # print(str(ans))
    return ans


def show_db_record(use_visual_output=True, doc=None):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''
    if doc is None:
        print('no doc specified')
        return

    for topic in doc:
        try: 
            print(str(topic) + ':' + str(doc[topic]))
        except UnicodeEncodeError:
            print('unicode encode error')
            print(topic.encode('utf-8') + ':' + doc[topic].encode('utf-8'))

    xlarge_url = doc['image']['sizes']['XLarge']['url']
    print('large img url:' + str(xlarge_url))
    if use_visual_output:
        img_arr = Utils.get_cv2_img_array(xlarge_url)
        if 'bounding_box' in doc:
            if Utils.legal_bounding_box(doc['bounding_box']):
                bb1 = doc['bounding_box']
                cv2.rectangle(img_arr, (bb1[0], bb1[1]), (bb1[0] + bb1[2], bb1[1] + bb1[3]), [255, 255, 0],
                              thickness=2)
        cv2.imshow('im1', img_arr)
        k = cv2.waitKey(200) & 0xFF


# lookfor_next_unbounded_feature_from_db_category got an unexpected keyword argument 'current_item'
    # answers = lookfor_next_unbounded_feature_from_db_category(current_item=current_item,
    #                                                                          skip_if_marked_to_skip=skip_if_marked_to_skip,
    #                                                                          which_to_show=which_to_show,
    #                                                                          filter_type=filter_type,
    #                                                                          category_id=category_id,
    #                                                                         word_in_description=word_in_description)

# structure of 'feature_bbs':
# feature_bbs:{'skip':True/False,'byCategoryId':{catID:[bb11,bb12,bb13],catID2:[bb21,bb22,],catID3:[bb3],...},'byWordInDescription':{word:[bb],word2:[b2],..}
def get_first_qualifying_record(cursor, which_to_show='showAll', filter_type='byCategoryId', filter=None, item_number=0,
                                skip_if_marked_to_skip=True):
    TOTAL_PRODUCTS = cursor.count()
    # print('tot records found:' + str(TOTAL_PRODUCTS))
    # make sure theres a known which_to_show

    # ['onlyShowBoxed', 'onlyShowUnboxed', 'showAll']:

    # if showAll, skip if necessary , otherwise return doc+bb if bb found, just doc otherwise
    if which_to_show == 'showAll':
        for i in range(item_number, TOTAL_PRODUCTS):
            doc = cursor[i]
            if 'feature_bbs' in doc:
                bbs = doc['feature_bbs']
                if skip_if_marked_to_skip:
                    if 'skip' in bbs:
                        if bbs['skip'] == True:
                            continue
                if filter_type in bbs:
                    bbs_of_filter_type = bbs[filter_type]
                    if filter in bbs_of_filter_type:
                        bbs = bbs_of_filter_type[filter]
                        if bbs is not None:
                            return {'doc': doc, 'bbs': bbs}
            return {'doc': doc}

    # if showBoxed, skip if necessary , otherwise return doc+bb if bb found
    elif which_to_show == 'showBoxed':
        for i in range(item_number, TOTAL_PRODUCTS):
            doc = cursor[i]
            if 'feature_bbs' in doc:
                bbs = doc['feature_bbs']
                if skip_if_marked_to_skip:
                    if 'skip' in bbs:
                        if bbs['skip'] == True:
                            continue
                if filter_type in bbs:
                    bbs_of_filter_type = bbs[filter_type]
                    if filter in bbs_of_filter_type:
                        bbs = bbs_of_filter_type[filter]
                        if bbs is not None:
                            return {'doc': doc, 'bbs': bbs}
        print('didnt find any relevant bounded boxes ')
        return None

    # if showUnBoxed, skip if necessary , otherwise return doc if bb NOT found
    elif which_to_show == 'showUnboxed':
        for i in range(item_number, TOTAL_PRODUCTS):
            doc = cursor[i]
            if 'feature_bbs' in doc:
                bbs = doc['feature_bbs']
                if skip_if_marked_to_skip:
                    if 'skip' in bbs:
                        if bbs['skip'] == True:
                            continue
                if filter_type in bbs:
                    bbs_of_filter_type = bbs[filter_type]
                    if filter in bbs_of_filter_type:
                        bbs = bbs_of_filter_type[filter]
                        if bbs is not None:
                            continue
                    else:
                        return {'doc': doc}
                else:
                    return {'doc': doc}
            else:
                return {'doc': doc}
        print('didnt find any relevant unbounded docs ')
        return None
    else:
        logging.warning(
            'cant figure out what you want dude - neither showall,showBoxed, showUnboxed, you wanted:' + str(
                which_to_show))


def suits_for_kyle():
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, 'mens-suits')}}}}
    keyword = 'Marcus'
    cursor = db.products.find({'$and': [{"brandedName": {'$regex': keyword}}, query]})
    N = cursor.count()
    print('found ' + str(N) + ' items')
    i = 0
    total = 1000
    for item in cursor:
        if i > total:
            break
        i += 1
        url = item['image']['sizes']['Best']['url']
        print('url:' + url)
        item_image = Utils.get_cv2_img_array(item['image']['sizes']['Best']['url'])


def reconstruct_db_images(images_collection):
    coll = db[images_collection]
    docs_cursor = coll.find({'people.person_bb': {'$exists': 0}})
    print('starting reconstruction on {0} documents'.format(docs_cursor.count()))
    docs_cursor.rewind()
    i = 0
    for doc in docs_cursor:
        if i % 10 == 0:
            print('performing the {0}th doc'.format(i))
            i += 1
        try:
            image = Utils.get_cv2_img_array(doc['image_urls'][0])
            for person in doc['people']:
                if len(person['face']):
                    x, y, w, h = person['face']
                    person_bb = [int(round(max(0, x - 1.5 * w))), y, int(round(min(image.shape[1], x + 2.5 * w))),
                                 min(image.shape[0], 8 * h)]
                else:
                    person_bb = None
                coll.update_one({'people': {'$elemMatch': {'person_id': person['person_id']}}},
                                {'$set': {'people.$.person_bb': person_bb}}, upsert=True)
        except Exception as e:
            print(str(e))


            # description: classic neckline , round collar, round neck, crew neck, square neck, v-neck, clASsic neckline,round collar,crewneck,crew neck, scoopneck,square neck, bow collar, ribbed round neck,rollneck ,slash neck
# cats:[{u'shortName': u'V-Necks', u'localizedId': u'v-neck-sweaters', u'id': u'v-neck-sweaters', u'name': u'V-Neck Sweaters'}]
# cats:[{u'shortName': u'Turtlenecks', u'localizedId': u'turleneck-sweaters', u'id': u'turleneck-sweaters', u'name': u'Turtlenecks'}]
# cats:[{u'shortName': u'Crewnecks & Scoopnecks', u'localizedId': u'crewneck-sweaters', u'id': u'crewneck-sweaters', u'name': u'Crewnecks & Scoopnecks'}]
# categories:#            u'name': u'V-Neck Sweaters'}]

def generate_id():
    id = objectid.ObjectId()
    return id


def clean_duplicates(collection, field):
    collection = db[collection]
    before = collection.count()
    sorted = collection.find().sort(field, pymongo.ASCENDING)
    print('starting, total {0} docs'.format(before))
    current_url = ""
    i = deleted = 0
    for doc in sorted:
        i += 1
        if i % 1000 == 0:
            print("deleted {0} docs after running on {1}".format(deleted, i))
        if doc['image_urls'][0] != current_url:
            current_url = doc['image_urls'][0]
            deleted += collection.delete_many({'$and': [{'image_urls': doc['image_urls'][0]}, {'_id': {'$ne': doc['_id']}}]}).deleted_count

    print("total {0} docs were deleted".format(deleted))


def hash_all_products():

    for gender in ['Female', 'Male']:
        collection = db['ebay_' + gender]
        for doc in collection.find({'img_hash': {'$exists': 0}}):
            while hash_q.count > 10000:
                time.sleep(5)
            hash_q.enqueue_call(func=hash_the_image, args=(doc['_id'], doc['images']['XLarge'], collection.name))


def hash_the_image(id, image_url, collection):
        collection = db[collection]
        image = Utils.get_cv2_img_array(image_url)
        if image is not None:
            img_hash = page_results.get_hash(image)
            collection.update_one({'_id': id}, {'$set': {'img_hash': img_hash}})


def clean_duplicates_aggregate(collection, key):
    pipeline = [{"$group": {"_id": "$"+key, "dups": {'$addToSet': "$_id"}, "count": {"$sum": 1}}}]
    for group in list(db[collection].aggregate(pipeline)):
        first = True
        if group['count'] > 1:
            for dup in group['dups']:
                if first:
                    first = False
                    continue
                else:
                    db[collection].delete_one({'_id': ObjectId(dup)})


def rebuild_similar_results():
    i = 0
    print("gonna do {0} images".format(db.images.count()))
    for image_obj in db.images.find():
        i += 1
        if i % 100 == 0:
            print("done {0} images".format(i))
        domain = tldextract.extract(image_obj['page_urls'][0]).registered_domain
        if domain in constants.products_per_site.keys():
            products_coll = constants.products_per_site[domain]
        else:
            products_coll = constants.products_per_site['default']
        try:
            for person in image_obj['people']:
                if 'items' in person.keys():
                    for item in person['items']:
                        if isinstance(item['similar_results'], list):
                            similar_dict = {products_coll: item['similar_results']}
                            item['similar_results'] = similar_dict
            db.images.replace_one({'_id': image_obj['_id']}, image_obj)
        except Exception as e:
            print(e)


def update_similar_results():
    i = 0
    for image_obj in db.images.find():
        i += 1
        print("done {0} images".format(i))
        for person in image_obj['people']:
            for item in person['items']:
                for collection in item['similar_results'].keys():
                    res_coll = collection + '_' + person['gender']
                    fp, item['similar_results'][collection] = find_top_n_results(number_of_results=100,
                                                                                 category_id=item['category'],
                                                                                 fingerprint=item['fp'],
                                                                                 collection=res_coll)
        res = db.images.replace_one({'_id': image_obj['_id']}, image_obj)
        if not res.modified_count:
            print(str(image_obj['_id']) + ' not inserted..')
    print("Done!!")


def add_sleeve_length_to_relevant_items_in_collection(col_name):
    rel_cats = set([cat for cat in constants.features_per_category.keys() if 'sleeve_length' in constants.features_per_category[cat]])
    collection = db[col_name]
    print("Starting, total {0} items".format(collection.count()))
    sent = 0
    for doc in collection.find():
        if col_name == 'images':
            doc_features = [item['fp'].keys() for person in doc['people'] for item in person['items']]
            flattened = [item for sublist in doc_features for item in sublist]
            if 'sleeve_length' in set(flattened):
                continue
            img_url = doc['image_urls'][0]

        else:
            category = doc['categories']
            if category not in rel_cats:
                print ('item not relevant for sleevedoll')
                continue
            fp = doc['fingerprint']
            if type(fp) != dict:
                fp = {'color':fp}
            doc_features = fp.keys()
            if 'sleeve_length' in doc_features:
                continue
            img_url = doc['images']['XLarge']

        item_id = doc['_id']
        add_feature.enqueue(parallel_sleeve_and_replace, args=(item_id, col_name, img_url), timeout=2000)
        sent += 1
        print('Sent {0} docs by now..'.format(sent))


def parallel_sleeve_and_replace(image_obj_id, col_name, img_url):
    rel_cats = set([cat for cat in constants.features_per_category.keys() if 'sleeve_length' in constants.features_per_category[cat]])
    collection = db[col_name]
    try:
        image = Utils.get_cv2_img_array(img_url)
        if image is None:
            collection.delete_one({'_id': image_obj_id})
            print("images deleted")
            return
        image_obj = collection.find_one({'_id': image_obj_id})
        if col_name == 'images':
            print("inside parallel replace")
            logging.warning('inside parallel replace')
            for person in image_obj['people']:
                person_cats = set([item['category'] for item in person['items']])
                if len(rel_cats.intersection(person_cats)):
                    print("there are matching categories")
                    if len(image_obj['people']) > 1:
                        image = background_removal.person_isolation(image, person['face'])
                    for item in person['items']:
                        if item['category'] in rel_cats:
                            sleeve_vector = [num.item() for num in sleeve_client.get_sleeve(image)['data']]
                            print("sleeve result: {0}".format(sleeve_vector))
                            item['fp']['sleeve_length'] = list(sleeve_vector)
            rep_res = db.images.replace_one({'_id': image_obj['_id']}, image_obj).modified_count
            print("{0} documents modified..".format(rep_res))
            return
        else:
            logging.warning('working on item from %s' %col_name)
            sleeve_vector = [num.item() for num in sleeve_client.get_sleeve(image)['data']]
            print("sleeve result: {0}".format(sleeve_vector))
            image_obj['fingerprint']['sleeve_length'] = list(sleeve_vector)
            rep_res = collection.replace_one({'_id': image_obj['_id']}, image_obj).modified_count
            print("{0} documents modified..".format(rep_res))
            return
    except Exception as e:
        print(e)


if __name__ == '__main__':
    print('starting')
    id = generate_id()
    print('id:' + str(id))
    # show_all_bbs_in_db()
    # fix_all_bbs_in_db()
    # doc = lookfor_next_unbounded_feature_from_db_category()
    # print('doc:' + str(doc))
    # suits_for_kyle()

    # step_thru_images_db(use_visual_output=True, collection='products')

    step_thru_db(use_visual_output=True, collection='products')
    # prune_training_db(use_visual_output=False)
    # lookfor_next_unbounded_feature_from_db_category(current_item=0, skip_if_marked_to_skip=False,
    # which_to_show='showAll', filter_type='byCategoryID',
    # category_id='dresses', word_in_description=None, db=None)