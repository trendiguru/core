__author__ = 'liorsabag'
import time
import csv
import gzip
import json
import numpy as np
import requests
from cv2 import imread, imdecode, imwrite
import logging
from bson import objectid
import pymongo
import os
from requests import ConnectionError

import constants

min_images_per_doc = constants.min_images_per_doc

# import urllib
# logging.setLevel(logging.DEBUG)

def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, try_url_locally=False, download=False,
                      download_directory='images'):
    """
    This function takes an url path and turn it to an image array
    :param url_or_path_to_image_file_or_cv2_image_array:
    :param try_url_locally:
    :param download:
    :param download_directory:
    :return: img_array
    """
    got_locally = False
    img_array = None  # attempt to deal with non-responding url
    # first check if we already have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, np.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array
    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        # try getting url locally by changing url to standard name
        if try_url_locally:  # turn url into local filename and try getting it again
            # FILENAME = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[0]
            FILENAME = \
                url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                    -1]  # jeremy changed this sinc it didnt work with url https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR2oSMcnwErH1eqf4k8fvn2bAxvSdDSbp6voC7ijYJStL2NfX6v
            FILENAME = os.path.join(download_directory, FILENAME)
            if FILENAME.endswith('jpg') or FILENAME.endswith('jpeg') or FILENAME.endswith('.bmp') or FILENAME.endswith(
                    'tiff'):
                pass
            else:  # there's no 'normal' filename ending so add .jpg
                FILENAME = FILENAME + '.jpg'
            # print('trying to use filename:'+str(FILENAME)+' and calling myself')
            img_array = get_cv2_img_array(FILENAME, try_url_locally=False, download=download,
                                          download_directory=download_directory)
            if img_array is not None:
                # print('got ok array calling self locally')
                return img_array
            else:  # couldnt get locally so try remotely
                # print('trying again since using local filename didnt work, download='+str(download))
                return (get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, try_url_locally=False,
                                          download=download, download_directory=download_directory))
        # put images in local directory
        else:
            # get remotely if its a url, get locally if not
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
                img_url = url_or_path_to_image_file_or_cv2_image_array
                try:
                    response = requests.get(img_url)  # download
                    img_array = imdecode(np.asarray(bytearray(response.content)), 1)
                except ConnectionError:
                    logging.warning("connection error - check url or connection")
                    return None
                except:
                    logging.warning("connection error - check url or connection")
                    return None

            else:  # get locally, since its not a url
                img_path = url_or_path_to_image_file_or_cv2_image_array
                try:
                    img_array = imread(img_path)
                    got_locally = True
                except:
                    logging.warning("connection error - check url or connection")
                    return None

    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, np.ndarray) and isinstance(img_array[0][0], np.ndarray)):
        logging.warning("Bad image - check url/path/array")
        return (None)
    # if we got good image and need to save locally :
    if download:
        if not got_locally:  # only download if we didn't get file locally
            if not os.path.isdir(download_directory):
                os.makedirs(download_directory)
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
                FILENAME = \
                    url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                        -1]
                FILENAME = os.path.join(download_directory, FILENAME)
            else:
                FILENAME = os.path.join(download_directory, url_or_path_to_image_file_or_cv2_image_array)
            if FILENAME.endswith('jpg') or FILENAME.endswith('jpeg') or FILENAME.endswith('.bmp') or FILENAME.endswith(
                    'tiff'):
                pass
            else:  # there's no 'normal' filename ending
                FILENAME = FILENAME + '.jpg'
            try:
                print('filename for local write:' + str(FILENAME))
                write_status = imwrite(FILENAME, img_array)
            except:
                print('unexpected error in Utils calling imwrite')
                # print('unexpected error in Utils calling urllib.urlretreive'+sys.exc_info()[0])

    return img_array


def count_human_bbs_in_doc(dict_of_images):
    n = 0
    for entry in dict_of_images:
        if good_bb(entry):
            n = n + 1  # got a good bb
    return (n)


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
                print('utils.py:image is marked to NOT be skipped')
                logging.debug('Utils.py(debug):image is marked to NOT be skipped')
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
        elif not (legal_bounding_box(entry["human_bb"])):
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


def lookfor_next_bounded_image(queryobject):
    """
    finds next image that has bounding box
    :param queryobject: this is a db entry
    :return:url, skip (whether or not to skip)
    """
    answers = {}
    n = 0
    skip_image = False
    got_unbounded_image = False
    urlN = None  # if nothing eventually is found None is returned for url
    if not 'images' in queryobject:
        logging.debug('Utils.py(debug):no images in input:' + str(queryobject))
        return None
    images = queryobject["images"]
    # print('utils.py:images:'+str(images))
    logging.debug('Utils.py(debug):images:' + str(images))
    # check for suitable number of images in doc - removed since i wanna check all the bbs
    # if len(images) < min_images_per_doc:  # don't use docs with too few images
    #        print('# images is too small:' + str(len(images)) + ' found and ' + str(min_images_per_doc) + ' are required')
    #        logging.debug('Utils.py(debug):image is marked to be skipped')
    #        return None
    print('# images:' + str(len(images)))
    try:
        answers["_id"] = str(queryobject["_id"])
    except KeyError, e:
        print 'keyerror on key "%s" which probably does not exist' % str(e)
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    try:
        answers["product_title"] = queryobject["Product Title"]
    except KeyError, e:
        print 'keyerror on key "%s" which probably does not exist' % str(e)
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    try:
        answers["product_url"] = queryobject["Product URL"]
    except KeyError, e:
        print 'keyerror on key "%s" which probably does not exist' % str(e)
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    try:
        id = queryobject['_id']
    except KeyError, e:
        print 'keyerror on key "%s" which probably does not exist' % str(e)
        logging.debug('keyerror on key "%s" which probably does not exist' % str(e))
    i = 0
    for entry in images:
        if 'skip_image' in entry:
            if entry['skip_image'] == True:
                print('utils.py:image is marked to be skipped')
                logging.debug('Utils.py(debug):image is marked to be skipped')
                skip_image = True
                answers['skip_image'] = True
            else:
                print('utils.py:image is NOT marked to be skipped')
                logging.debug('Utils.py(debug):image is NOT marked to be skipped')
                skip_image = False
                answers['skip_image'] = False

        if 'human_bb' in entry:  # got a pic with a bb
            print('utils.py:there is a human bb entry for:' + str(entry))
            answers['url'] = entry['url']
            answers['bb'] = entry['human_bb']
            answers['image_index'] = i
            return answers

        else:  # no human_bb in this entry
            i = i + 1

    else:
        print('utils.lookfor_next_bounded_image:no bounded image found in this doc:(')
        logging.debug('utils.lookfor_next_bounded_image - no bounded image found in this doc')
        return None


def lookfor_next_bounded_in_db(i=0):
    """
    find next bounded image in db
    :input: i, the index of the current item
    :return:url,bb, and skip_it for next unbounded image
    """
    db = pymongo.MongoClient().mydb
    # training docs contains lots of different images (URLs) of the same clothing item
    training_collection_cursor = db.training.find()   #.sort _id
#    doc = next(training_collection_cursor, None)
    doc = training_collection_cursor[0]
    i = 0
    while doc is not None:
        print('doc:' + str(doc))
        answers = lookfor_next_bounded_image(doc)
        if answers is not None:
            try:
                if answers["bb"] is not None:  # got a good bb
                    id = doc['_id']
#                    write_result = db.training.update({"_id": objectid.ObjectId(id)}, {"$set": {"images": doc['images']}} )
#                    write_result = db.training.update({"_id": objectid.ObjectId(id)}, {"$set": {"images": doc['images']}} )
                    return answers,i
            except KeyError, e:
                print 'hi there was a keyerror on key "%s" which probably does not exist' % str(e)
        i = i + 1
        doc = training_collection_cursor[i]
        logging.debug("no bounded image found in current doc, trying next")

    print("no bounded image found in collection")
    logging.debug("no bounded image found in collection")
    return "No bounded bb found in db"


def good_bb(dict):
    '''
    determine if dict has good human bb in it
    '''
    if not 'human_bb' in dict:
        # print('no human_bb key in dict')
        return (False)
    elif dict["human_bb"] is None:
        # print('human_bb is None')
        return (False)
    elif not (legal_bounding_box(dict["human_bb"])):
        # print('human bb is not big enough')
        return (False)
    else:
        # print('human bb ok:'+str(dict['human_bb']))
        return (dict["human_bb"])


def legal_bounding_box(rect):
    minimum_allowed_area = 50
    if rect[2] * rect[3] >= minimum_allowed_area:
        return True
    else:
        return False


def check_img_array(image_array):
    if image_array is not None and isinstance(image_array, np.ndarray) and isinstance(image_array[0][0], np.ndarray):
        return True

    else:
        return False


def bounding_box_inside_image(image_array, rect):
    if check_img_array(image_array) and legal_bounding_box(rect):
        height, width, depth = image_array.shape
        if rect[2] <= width and rect[3] <= height:
            return True  # bb fits into image
        else:
            return False
    else:
        return False


# test function for lookfor_next_unbounded_image
def test_lookfor_next():
    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()  # The db with multiple figs of same item
    doc = next(training_collection_cursor, None)
    resultDict = {}
    while doc is not None:
        if url:
            resultDict["url"] = url
            resultDict["_id"] = str(doc['_id'])
            # a better way to deal with keys that may not exist;
            try:
                resultDict["product_title"] = doc["Product Title"]
            except KeyError, e:
                print 'hi there was a keyerror on key "%s" which probably does not exist' % str(e)
            try:
                resultDict["product_url"] = doc["Product URL"]
            except KeyError, e:
                print 'hi there was a keyerror on key "%s" which probably does not exist' % str(e)
            return resultDict
        else:
            print("no unbounded image found for string:" + str(prefix) + " in current doc")
            logging.debug("no unbounded image found for string:" + str(prefix) + " in current doc")
        doc = next(training_collection_cursor, None)
    return resultDict


# products_collection_cursor = db.products.find()   #Regular db of one fig per item

# prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
#training docs contains lots of different images (URLs) of the same clothing item
#logging.debug(str(doc))
#print('doc:'+str(doc))
#       for prefix in prefixes:


def test_count_bbs():
    '''
    test counting how many good bb;s in doc
    '''

    db = pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()  # The db with multiple figs of same item
    doc = next(training_collection_cursor, None)
    resultDict = {}
    while doc is not None:
        if 'images' in doc:
            n = count_human_bbs_in_doc(doc['images'])
            print('number of good bbs:' + str(n))
        doc = next(training_collection_cursor, None)


def test_insert_bb(dict, bb):
    db = pymongo.MongoClient().mydb
    doc = db.good_training_set.find_one({'_id': objectid.ObjectId(dict['_id'])})
    imagelist = doc['images']
    print('imagelist:' + str(imagelist))
    for item in imagelist:
        print('item:' + str(item))
        print('desired url:' + str(dict['url']) + 'actual item url:' + str(item['url']))
        if item['url'] == dict['url']:  # this is the right image
            print('MATCH')
            item['human_bb'] = bb
            print('imagelist after bb insertion:' + str(imagelist))
            db.good_training_set.update({"_id": objectid.ObjectId(dict["_id"])}, {'$set': {'images': imagelist}})
            return True


def test_lookfor_and_insert():
    dict = test_lookfor_next()
    test_insert_bb(dict, [10, 20, 30, 40])


class GZipCSVReader:
    def __init__(self, filename):
        self.gzfile = gzip.open(filename)
        self.reader = csv.DictReader(self.gzfile)

    def next(self):
        return self.reader.next()

    def close(self):
        self.gzfile.close()

    def __iter__(self):
        return self.reader.__iter__()


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
