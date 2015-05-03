from __future__ import print_function

import multiprocessing

__author__ = 'liorsabag'
import csv
import gzip
import json
import requests
from cv2 import imread, imdecode, imwrite
import logging
import os
from requests import ConnectionError
import time
import numpy as np
from bson import objectid
import pymongo
import constants
import math
import cv2
import re

min_images_per_doc = constants.min_images_per_doc
max_image_val = constants.max_image_val

# import urllib
# logging.setLevel(logging.DEBUG)

def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False, download=False,
                      download_directory='images'):
    """
    Get a cv2 img array from a number of different possible inputs.

    :param url_or_path_to_image_file_or_cv2_image_array:
    :param convert_url_to_local_filename:
    :param download:
    :param download_directory:
    :return: img_array
    """
    # print('get:' + str(url_or_path_to_image_file_or_cv2_image_array) + ' try local' + str(
    # convert_url_to_local_filename) + ' download:' + str(download))
    got_locally = False
    img_array = None  # attempt to deal with non-responding url
    # first check if we already have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, np.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array

    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        # try getting url locally by changing url to standard name
        if convert_url_to_local_filename:  # turn url into local filename and try getting it again
            # filename = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[0]
            # jeremy changed this since it didn't work with url -
            # https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR2oSMcnwErH1eqf4k8fvn2bAxvSdDSbp6voC7ijYJStL2NfX6v
            #TODO: find a better way to create legal filename from url
            filename = \
                url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                    -1]
            filename = os.path.join(download_directory, filename)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or \
                    filename.endswith('tiff'):
                pass
            else:  # there's no 'normal' filename ending so add .jpg
                filename = filename + '.jpg'
            # print('trying again locally using filename:' + str(filename))
            img_array = get_cv2_img_array(filename, convert_url_to_local_filename=False, download=download,
                                          download_directory=download_directory)
            #maybe return(get_cv2 etc) instead of img_array =
            if img_array is not None:
                # print('got ok array calling self locally')
                return img_array
            else:  # couldnt get locally so try remotely
                # print('trying again remotely since using local filename didnt work, download=' + str( download) + ' fname:' + str(filename))
                return (
                    get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False,
                                      download=download,
                                      download_directory=download_directory))  # this used to be 'return'
        # put images in local directory
        else:
            # get remotely if its a url, get locally if not
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
                img_url = url_or_path_to_image_file_or_cv2_image_array
                try:
                    # print("trying remotely (url) ")
                    response = requests.get(img_url)  # download
                    img_array = imdecode(np.asarray(bytearray(response.content)), 1)
                except ConnectionError:
                    logging.warning("connection error - check url or connection")
                    return None
                except:
                    logging.warning(" error other than connection error - check something other than connection")
                    return None

            else:  # get locally, since its not a url
                # print("trying locally (not url)")
                img_path = url_or_path_to_image_file_or_cv2_image_array
                try:
                    img_array = imread(img_path)
                    if img_array is not None:
                        # print("success trying locally (not url)")
                        got_locally = True
                    else:
                        # print('couldnt get locally (in not url branch)')
                        return None
                except:
                    # print("could not read locally, returning None")
                    logging.warning("could not read locally, returning None")
                    return None  # input isn't a basestring nor a np.ndarray....so what is it?
    else:
        logging.warning("input is neither an ndarray nor a string, so I don't know what to do")
        return None

    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, np.ndarray) and isinstance(img_array[0][0], np.ndarray)):
        print("Bad image - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        logging.warning("Bad image - check url/path/array:" + str(
            url_or_path_to_image_file_or_cv2_image_array) + 'try locally' + str(
            convert_url_to_local_filename) + ' dl:' + str(
            download) + ' dir:' + str(download_directory))
        return (None)
    # if we got good image and need to save locally :
    if download:
        if not got_locally:  # only download if we didn't get file locally
            if not os.path.isdir(download_directory):
                os.makedirs(download_directory)
            if "://" in url_or_path_to_image_file_or_cv2_image_array:  # its a url, get the bifnocho
                filename = \
                    url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[
                        -1]
                filename = os.path.join(download_directory, filename)
            else:  # its not a url so use straight
                filename = os.path.join(download_directory, url_or_path_to_image_file_or_cv2_image_array)
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('.bmp') or filename.endswith(
                    'tiff'):
                pass
            else:  # there's no 'normal' filename ending
                filename = filename + '.jpg'
            try:
                # print('filename for local write:' + str(filename))
                write_status = imwrite(filename, img_array)
                max_i = 50  # wait until file is readable before continuing
                for i in xrange(max_i):
                    try:
                        with open(filename, 'rb') as _:
                            break
                    except IOError:
                        time.sleep(10)
                    else:
                        print('Could not access {} after {} attempts'.format(filename, str(max_i)))
                        raise IOError('Could not access {} after {} attempts'.format(filename, str(max_i)))
            except:  # this is prob unneeded given the 'else' above
                print('unexpected error in Utils calling imwrite')
    return img_array

def count_human_bbs_in_doc(dict_of_images, skip_if_marked_to_skip=True):
    n = 0
    for entry in dict_of_images:
        print('entry:' + str(entry) + ' n=' + str(n), end='\r')
        if good_bb(entry, skip_if_marked_to_skip=skip_if_marked_to_skip):
            n = n + 1  # dont care if marked to be skipped
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
    #        return None
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
    if image_index == max_image_val:  #max_image_val is a code meaning get the last image
        image_index = len(images) - 1
    if image_index >= len(images):  #index starts at 0
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
            #            answers['x'] = entry['human_bb'][0]
            #            answers['y'] = entry['human_bb'][1]
            #            answers['w'] = entry['human_bb'][2]
            #            answers['h'] = entry['human_bb'][3]
            return answers

        elif only_get_boxed_images == False:  # no human_bb in this entry but its ok, return anyway
            return answers

    print('utils.lookfor_next_bounded_image:no bounded image found in this doc:(')
    logging.debug('utils.lookfor_next_bounded_image - no bounded image found in this doc')
    return None

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
        #	logging.warning('calling lookfor_next_bounded, index='+str(i)+' image='+str(current_image))
        answers = lookfor_next_bounded_image(doc, image_index=current_image,
                                             only_get_boxed_images=only_get_boxed_images)
        #	logging.warning('returned from  lookfor_next_bounded')
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
                    #go to next doc since no bb was found in this one
            else:
                logging.debug('exiting lookfornext 2, answers:' + str(answers))
                return answers
        i = i + 1
        current_image = 0
        doc = training_collection_cursor[i]
        logging.warning("no bounded image found in current doc, trying next")
    return {'error': 0, 'message': "No bounded bb found in db"}

def good_bb(dict, skip_if_marked_to_skip=True):
    '''
    determine if dict has good human bb in it
    '''

    if skip_if_marked_to_skip:
        if "skip_image" in dict:
            if dict['skip_image'] == True:
                return (False)

    if not 'url' in dict:
        # print('img is none')
        return (False)

    url = dict['url']
    img_arr = get_cv2_img_array(url, convert_url_to_local_filename=True, download=True,
                                download_directory='images')
    if not is_valid_image(img_arr):
        print('bad image array discovered in is_valid_image')
        return False
    if not 'human_bb' in dict:
        # print('no human_bb key in dict')
        return (False)
    if dict["human_bb"] is None:
        # print('human_bb is None')
        return (False)
    bb = dict['human_bb']
    if not bounding_box_inside_image(img_arr, bb):  #
        print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
            url))
        return (False)

    return (True)

def legal_bounding_box(rect):
    if rect is None:
        return False
    minimum_allowed_area = constants.min_image_area
    if rect[2] * rect[3] >= minimum_allowed_area:
        return True
    else:
        print('area of ' + str(rect[2]) + 'x' + str(rect[3]) + ':' + str(rect[2] * rect[3]))
        return False

def bounding_box_inside_image(image_array, rect):
    # if check_img_array(image_array) and legal_bounding_box(rect):
    if legal_bounding_box(rect):
        height, width = image_array.shape[0:2]
        if rect[0] < width and rect[0] + rect[2] <= width and rect[1] < height and rect[1] + rect[3] <= height:
            return True  # bb fits into image
        else:
            print('warning - bb not inside image')
            return False
    else:
        print('warning - bb not legal (either too small or None')
        return False


# products_collection_cursor = db.products.find()   #Regular db of one fig per item

# prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
# training docs contains lots of different images (URLs) of the same clothing item
# logging.debug(str(doc))
#print('doc:'+str(doc))
#       for prefix in prefixes:

def fix_all_bbs_in_db(use_visual_output=False):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''
    print('opening db')
    db = pymongo.MongoClient().mydb
    print('db open')
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
            img_arr = get_cv2_img_array(image_url, convert_url_to_local_filename=True, download=True,
                                        download_directory='images')
            if img_arr is None:
                print('img is none')
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
                if not bounding_box_inside_image(img_arr, bb):
                    print('bad bb caught,bb:' + str(bb) + ' img size:' + str(img_arr.shape) + ' imagedoc:' + str(
                        image) + ' h,w:' + str(height) + ',' + str(width))
                    print('h,w:' + str(height) + ',' + str(width))
                    if not legal_bounding_box(bb):  # too small, make right and bottom at edge of  image
                        print('not legal bounding box')
                        raw_input('not a legal bb...')
                        bb[2] = width - bb[0]
                        bb[3] = height - bb[1]
                    bb[0] = max(0, bb[0])  # if less than zero
                    bb[0] = min(bb[0], width - 1)  # if greater than width
                    bb[2] = max(0, bb[2])  # if less than 0
                    bb[2] = min(bb[2], width - bb[0] - 1)  # the -1 is just to make sure, prob unneeded

                    bb[1] = max(0, bb[1])  # if less than zero
                    bb[1] = min(bb[1], height - 1)  # if greater than height
                    bb[3] = max(0, bb[3])  # if less than zero
                    bb[3] = min(bb[3], height - bb[1] - 1)  # the -1 is just to make sure, prob unneeded
                    print ('suggested replacement:' + str(bb))
                    raw_input('got one')
                    image["human_bb"] = bb
                    id = str(doc['_id'])
                    write_result = db.training.update({"_id": objectid.ObjectId(id)},
                                                      {"$set": {"images": doc['images']}})
                    # TODO: check error on updating
                else:
                    print('not es tot', end='\r', sep='')
                    print('got good bb, i=' + str(i), end='\r', sep='')

        j = j + 1
        doc = next(training_collection_cursor, None)

    return {"success": 1}



class GZipCSVReader:
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
        #or can do it by looking for item # lke this
        #training_collection_cursor = db.training.find()   #.sort _id
        #doc = training_collection_cursor[current_item]
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
    def __init__(self, filename):
        self.gzfile = gzip.open(filename)
        self.reader = csv.DictReader(self.gzfile)

    def next(self):
        return self.reader.next()

    def close(self):
        self.gzfile.close()

    def __iter__(self):
        return self.reader.__iter__()


class npAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ThreadSafeCounter(object):
    def __init__(self):
        self.val = multiprocessing.Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

def bb_to_mask(bb, img_array):
    '''
    bb in form of x,y,w,h converted to np array the same size as img_array
    :param bb:
    :return:
    '''
    h, w = img_array.shape[0:2]
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    if bounding_box_inside_image(img_array, bb):
        mask[bb[0]:(bb[0] + bb[2]), bb[1]:(bb[1] + bb[3])] = 1
    elif bb[0] + bb[2] <= w and bb[1] + bb[3] <= h:  # left and top edges are ok
        mask[bb[0]:min(bb[0] + bb[2], w), bb[1]:min(bb[1] + bb[3], h)] = 1
    else:  # left or top edge not ok so use entire box
        mask = np.ones((h, w), dtype=np.uint8)
    if mask.shape[0] != img_array.shape[0] or mask.shape[1] != img_array.shape[1]:
        print('trouble with mask size in bb_to_mask, resetting to image size')
        mask = np.ones((h, w), dtype=np.uint8)

    return mask


def is_valid_image(img):
    if img is not None and type(img) == np.ndarray and isinstance(img[0][0], np.ndarray) and img.shape[0] * img.shape[
        1] >= constants.min_image_area:
        return True
    else:
        return False

############################
### math stuff
############################


def error_of_fraction(numerator, numerator_stdev, denominator, denominator_stdev):
    """
    this gives the error on fraction numerator/denominator assuming no covariance
    :param numerator:
    :param numerator_stdev:
    :param denominator:
    :param denominator_stdev:
    :return:
    """
    n = float(numerator)
    d = float(denominator)
    n_e = float(numerator_stdev)
    d_e = float(denominator_stdev)
    if n == 0 or d == 0:
        print('caught div by zero in error_of_fraction, n=' + str(n) + ' d=' + str(d))
        return (-1.0)
    fraction_error = abs(n / d) * math.sqrt((n_e / n) ** 2 + (d_e / d) ** 2)
    return fraction_error


def isnumber(str):
    num_format = re.compile("^[1-9][0-9]*\.?[0-9]*")
    isnumber = re.match(num_format, str)
    if isnumber:
        return True
    else:
        return False




if __name__ == '__main__':
    print('starting')
    fix_all_bbs_in_db()