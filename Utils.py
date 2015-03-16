__author__ = 'liorsabag'
import time
import csv
import gzip
import json
import numpy
import requests
from cv2 import imread, imdecode
import logging
from bson import objectid
import pymongo

#logging.setLevel(logging.DEBUG)

def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array):
    # first check if we have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, numpy.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array
    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
        if "://" in url_or_path_to_image_file_or_cv2_image_array:
            img_url = url_or_path_to_image_file_or_cv2_image_array
            img_array = imdecode(numpy.asarray(bytearray(requests.get(img_url).content)), 1)
        else:
            img_path = url_or_path_to_image_file_or_cv2_image_array
            img_array = imread(img_path)
    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, numpy.ndarray) and isinstance(img_array[0][0], numpy.ndarray)):
        logging.warning("Bad image - check url/path/array")
    return img_array

def count_bbs_in_doc(doc):
    n=0
    images = queryobject["images"]
    logging.debug('Utils.py(debug):images:'+str(images))
    for entry in images:
	if good_bb(entry):
		n=n+1   #got a good bb
    return(n)


def lookfor_next_unbounded_image(queryobject):
    min_images_per_doc = 10
    n=0
    got_unbounded_image = False
    urlN=None   #if nothing eventually is found None is returned for url
    images = queryobject["images"]
    #print('utils.py:images:'+str(images))
    logging.debug('Utils.py(debug):images:'+str(images))
    if len(images)<min_images_per_doc:   #don't use docs with too few images
	return(None)    
    print('# images:'+str(len(images)))
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
	    urlN=entry['url']
 	    got_unbounded_image = True
	    print('utils.py:no human bb entry for:'+str(entry))
	    return(urlN)
	elif entry["human_bb"] is None:
	    urlN=entry['url']
	    got_unbounded_image = True
	    print('utils.py:human_bb is None for:'+str(entry))
	    return(urlN)
    	elif not isinstance(entry["human_bb"],list):
	    urlN=entry['url']
	    got_unbounded_image = True
	    print('utils.py:illegal bb!! (not a list) for:'+str(entry))
	    return(urlN)		    
	elif not(legal_bounding_box(entry["human_bb"])):
	    urlN=entry['url']
	    got_unbounded_image = True
	    print('utils.py:bb is not legal (too small) for:'+str(entry))
	    return(urlN)
 	else:
	    urlN=None
	    got_unbounded_image = False
	    print('utils.py:image is bounded :(')
            logging.debug('image is bounded.....')
    return(urlN)
# maybe return(urlN,n) at some point


def good_bb(dict):
    '''
    determine if dict has good human bb in it
    '''
    if not 'human_bb' in dict:  # got a pic without a bb
	print('no human_bb key in dict')
	return(False)
    elif dict["human_bb"] is None:
	print('human_bb is None')
	return(False)
    elif not(legal_bounding_box(dict["human_bb"])):
	print('human bb is not big enough')
	return(False)
    else:
	print('human bb ok:'+str(dict['human_bb']))
	return(dict["human_bb"])

def legal_bounding_box(rect):
    minimum_allowed_area = 50
    if rect[2]*rect[3] >= minimum_allowed_area:
    	return True
    else:
	return False

#test function for lookfor_next_unbounded_image
def test_lookfor_next():
    db=pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()   #The db with multiple figs of same item
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
            print("no unbounded image found for string:" + str(prefix)+" in current doc")
            logging.debug("no unbounded image found for string:" + str(prefix)+ " in current doc")
    	doc = next(training_collection_cursor, None)
    return resultDict



#products_collection_cursor = db.products.find()   #Regular db of one fig per item

#    prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
            #training docs contains lots of different images (URLs) of the same clothing item
    	#logging.debug(str(doc))
        #print('doc:'+str(doc))
 #       for prefix in prefixes:


def test_count_bbs():
    '''
    test counting how many good bb;s in doc
    '''

    db=pymongo.MongoClient().mydb
    training_collection_cursor = db.good_training_set.find()   #The db with multiple figs of same item
    doc = next(training_collection_cursor, None)
    resultDict = {}
    while doc is not None:
	count_bbs_in_doc(doc)
	print('number of good bbs')
    	doc = next(training_collection_cursor, None)


def test_insert_bb(dict,bb):
    db=pymongo.MongoClient().mydb
    doc = db.good_training_set.find_one({ '_id': objectid.ObjectId(dict['_id']) }) 
    imagelist = doc['images']
    print('imagelist:'+str(imagelist))
    for item in imagelist:
    	print('item:'+str(item))
        print('desired url:'+str(dict['url'])+'actual item url:'+str(item['url']))
        if item['url'] == dict['url']:  #this is the right image
        	print('MATCH')
                item['human_bb'] = bb
                print('imagelist after bb insertion:'+str(imagelist))
    	 	db.good_training_set.update({"_id":objectid.ObjectId(dict["_id"])}, {'$set':{'images':imagelist}})
    		return True

def test_lookfor_and_insert():
    dict = test_lookfor_next()
    test_insert_bb(dict,[10,20,30,40])

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
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
