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
import string


# import urllib
# logging.setLevel(logging.DEBUG)


def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.

Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.

"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename

def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, convert_url_to_local_filename=False, download=False,
                      download_directory='images', filename=False):
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
            try:  # write file then open it
                # print('filename for local write:' + str(filename))
                write_status = imwrite(filename, img_array)
                max_i = 50  # wait until file is readable before continuing
                gotfile = False
                for i in xrange(max_i):
                    try:
                        with open(filename, 'rb') as _:
                            gotfile = True
                    except IOError:
                        time.sleep(10)
                if gotfile == False:
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
    if all_inclusive_bounding_box(img_arr, bb):
        dict['human_bb'] = reduce_bounding_box(bb)  # attempting to avoid bbsize=imgsize
    return (True)

def legal_bounding_box(rect):
    if rect is None:
        return False
    minimum_allowed_area = constants.min_image_area
    if rect[2] * rect[3] < minimum_allowed_area:
        print('area of ' + str(rect[2]) + 'x' + str(rect[3]) + ':' + str(rect[2] * rect[3]))
        return False
    if rect[0] < 0 or rect[1] < 0 or rect[2] < 0 or rect[3] < 0:
        return False
    return True


def bounding_box_inside_image(image_array, rect):
    # if check_img_array(image_array) and legal_bounding_box(rect):
    if legal_bounding_box(rect):
        height, width = image_array.shape[0:2]
        if rect[0] < width and rect[0] + rect[2] < width and rect[1] < height and rect[1] + rect[3] < height:
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


def step_thru_db(use_visual_output=True, collection='products'):
    '''
    fix all the bbs so they fit their respective image
    :return:
    '''
    print('opening db')
    db = pymongo.MongoClient().mydb
    print('db open')
    if db is None:
        print('couldnt open db')
        return {"success": 0, "error": "could not get db"}
    dbstring = 'db.' + collection
    # cursor = dbstring.find()   look in defaults.py how this is done
    cursor = db.products.find()
    print('returned cursor')
    if cursor is None:  # make sure training collection exists
        print('couldnt get cursor ' + str(collection))
        return {"success": 0, "error": "could not get colelction"}
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

        large_url = doc['image']['sizes']['Large']['url']
        print('large img url:' + str(large_url))
        if use_visual_output:
            img_arr = get_cv2_img_array(large_url)
            if 'bounding_box' in doc:
                if legal_bounding_box(doc['bounding_box']):
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
            if not is_valid_image(img_arr):
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
        print(doc)
        i = 0
        for image in images:
            image_url = image["url"]
            if 'skip_image' in image:
                if image['skip_image'] == True:
                    print('marked for skip:' + str(i), end='\r')
                    continue
            img_arr = get_cv2_img_array(image_url, convert_url_to_local_filename=True, download=True,
                                        download_directory='images')
            if not is_valid_image(img_arr):
                print('img is not valid (=None or too small')
                continue

            if 'human_bb' in image:
                i = i + 1
                height, width = img_arr.shape[0:2]
                bb = image["human_bb"]
                if bb is None:
                    print('bb is None')
                    continue

                if not bounding_box_inside_image(img_arr, bb):
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


def all_inclusive_bounding_box(image_array, bounding_box):
    """
    determine if the bb takes up all or  almost all the image
    :param image_array:
    :param bounding_box:
    :return:whether the bb takes up almost all image (True) or not (False)
    """
    height, width = image_array.shape[0:2]
    image_area = float(height * width)
    bb_area = bounding_box[2] * bounding_box[3]
    if bb_area > constants.min_bb_to_image_area_ratio * image_area:
        # print('got a bb that takes nearly all image')
        # logging.warning('got a bb that takes nearly all image')
        return True
    else:
        return False


def reduce_bounding_box(bounding_box):
    """
    determine if the bb takes up all or  almost all the image
    :param bounding_box:
    :return:smaller bb (again attempting to get around grabcut bug )
    """
    newx = bounding_box[0] + 1
    new_width = bounding_box[2] - 1
    newy = bounding_box[1] + 1
    new_height = bounding_box[3] - 1
    newbb = [newx, newy, new_width, new_height]
    if legal_bounding_box(newbb):
        return newbb
    else:
        logging.warning('cant decrease size of bb')
        return bounding_box


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
    if img is not None and type(img) == np.ndarray and img.shape[0] * img.shape[
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
    #show_all_bbs_in_db()
    #fix_all_bbs_in_db()
    step_thru_db(use_visual_output=True)