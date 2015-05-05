__author__ = 'jeremy'

# builtin
import logging

import pymongo
from bson import objectid


# ours
import constants

import Utils

min_images_per_doc = constants.min_images_per_doc
max_image_val = constants.max_image_val


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
    db = pymongo.MongoClient().mydb
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
    db = pymongo.MongoClient().mydb
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
    db = pymongo.MongoClient().mydb
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

