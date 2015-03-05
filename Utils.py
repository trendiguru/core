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


def lookfor_next_unbounded_image(queryobject,string):
    n=0
    got_unbounded_image = False
    urlN=None   #if nothing eventually is found None is returned for url
    while got_unbounded_image is False:
    	n=n+1
	strN=string+str(n)  #this is to build strings like 'Main Image URL angle 5' or 'Style Gallery Image 7'
	bbN = strN+' bb' #this builds strings like 'Main Image URL angle 5 bb' or 'Style Gallery Image 7 bb'
	print('looking for string:'+str(strN)+' and bb '+str(bbN))
	logging.debug('looking for string:'+str(strN)+' and bb '+str(bbN))
	if strN in queryobject:
		if not 'human_bb' in queryobject:  # got a pic without a bb
			urlN=queryobject[strN]
 			got_unbounded_image = True
			print('image from string:'+strN+' :is not bounded!!')
		elif queryobject[humanbb] is None:
			urlN=queryobject[strN]
			got_unbounded_image = True
			print('image from string:'+strN+' :is not bounded!!')
 		else:
			urlN=None
			got_unbounded_image = False
			print('image from string:'+strN+' :is bounded :(')
	else:
		print('didn\'t find expected string in training db')
		logging.debug('didn\'t find expected string in training db')
		break
    return(urlN)
# maybe return(urlN,n) at some point


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