__author__ = 'liorsabag'
import csv
import gzip
import json
import numpy
import requests
from cv2 import imread, imdecode
import logging


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

# this is for the training collection, where there's a set of images from different angles in each record


def lookfor_next_unbounded_image(queryobject):
    n=0
    got_unbounded_image = False
    urlN=None   #if nothing eventually is found None is returned for url
    images = queryobject["images"]
    print('images:'+str(images))
    for entry in images:
    	print('entry:'+str(entry))
#    	n=n+1
#	strN=string+str(n)  #this is to build strings like 'Main Image URL angle 5' or 'Style Gallery Image 7'
#	bbN = strN+' bb' #this builds strings like 'Main Image URL angle 5 bb' or 'Style Gallery Image 7 bb'
#	print('entry:'+str(entry))
#	print('looking for string:'+str(strN)+' and bb '+str(bbN))
#	logging.debug('looking for string:'+str(strN)+' and bb '+str(bbN))
    	
#	if entry["old_name"] == strN:
#		print('found old_name:'+entry["old_name"])
    	if not 'human_bb' in entry:  # got a pic without a bb
	    urlN=queryobject[strN]
 	    got_unbounded_image = True
	    print('image from string:'+strN+' :is not bounded!!')
	elif entry["human_bb"] is None:
	    urlN=queryobject[strN]
	    got_unbounded_image = True
	    print('image from string:'+strN+' :is not bounded!!')
 	else:
	    urlN=None
	    got_unbounded_image = False
	    print('image is bounded :(')
    return(urlN)
# maybe return(urlN,n) at some point

#test function for lookfor_next_unbounded_image
import pymongo
def test_lookfor_next():
    db=pymongo.MongoClient().mydb
    training_collection_cursor = db.training.find()   #The db with multiple figs of same item
#products_collection_cursor = db.products.find()   #Regular db of one fig per item

#    prefixes = ['Main Image URL angle ', 'Style Gallery Image ']
    doc = next(training_collection_cursor, None)
    resultDict = {}
    while doc is not None:
            #training docs contains lots of different images (URLs) of the same clothing item
    	logging.debug(str(doc))
        print('doc:'+str(doc))
 #       for prefix in prefixes:
        url = lookfor_next_unbounded_image(doc)
        if url:
       	    resultDict["url"] = url
            resultDict["_id"] = str(doc['_id'])
    	    print('resultDict:'+str(resultDict));
            return resultDict
        else:
            print("no unbounded image found for string:" + str(prefix)+" in current doc")
            logging.debug("no unbounded image found for string:" + str(prefix)+ " in current doc")
    return resultDict


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
