__author__ = 'liorsabag'
import time
import csv
import gzip
import json
import numpy
import requests
from cv2 import imread, imdecode, imwrite
import logging
from bson import objectid
import pymongo
import os
#import urllib 

#logging.setLevel(logging.DEBUG)

def get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array, try_url_locally = False, download = False, download_directory = 'images'):
    got_locally = False
    img_array = None  #attempt to deal with non-responding url
    # first check if we already have a numpy array
    if isinstance(url_or_path_to_image_file_or_cv2_image_array, numpy.ndarray):
        img_array = url_or_path_to_image_file_or_cv2_image_array
    # otherwise it's probably a string, check what kind
    elif isinstance(url_or_path_to_image_file_or_cv2_image_array, basestring):
	#try getting url locally by changing url to standard name
	if try_url_locally: #turn url into local filename and try getting it again
#   	 	FILENAME = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[0]
   	 	FILENAME = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[-1]  #jeremy changed this sinc it didnt work with url https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcR2oSMcnwErH1eqf4k8fvn2bAxvSdDSbp6voC7ijYJStL2NfX6v
    		FILENAME = os.path.join(download_directory,FILENAME)
		if FILENAME.endswith('jpg')or FILENAME.endswith('jpeg') or FILENAME.endswith('.bmp') or FILENAME.endswith('tiff'):
			pass
		else:    #there's no 'normal' filename ending so add .jpg
			FILENAME=FILENAME+'.jpg' 
		print('trying to use filename:'+str(FILENAME)+' and calling myself')
		img_array = get_cv2_img_array(FILENAME,try_url_locally=False,download=download,download_directory=download_directory) 
		if img_array is not None:
			print('got ok array calling self locally')
			return img_array
		else:   #couldnt get locally so try remotely
			print('trying again since using local filename didnt work, download='+str(download))
			return (get_cv2_img_array(url_or_path_to_image_file_or_cv2_image_array,try_url_locally=False,download=download,download_directory=download_directory))
    	# put images in local directory
	else:    
#get remotely if its a url, get locally if not
            if "://" in url_or_path_to_image_file_or_cv2_image_array:
            	img_url = url_or_path_to_image_file_or_cv2_image_array
	    	try:
			response = requests.get(img_url)  #download 
	        	img_array = imdecode(numpy.asarray(bytearray(response.content)), 1)
	    	except ConnectionError:
         		logging.warning("connection error - check url or connection")
			return None
		except:
         		logging.warning("connection error - check url or connection")
			return None
			
            else:   #get locally, since its not a url
            	img_path = url_or_path_to_image_file_or_cv2_image_array
		try:
	            	img_array = imread(img_path)
			got_locally = True
		except:
         		logging.warning("connection error - check url or connection")
			return None

    # After we're done with all the above, this should be true - final check that we're outputting a good array
    if not (isinstance(img_array, numpy.ndarray) and isinstance(img_array[0][0], numpy.ndarray)):
        logging.warning("Bad image - check url/path/array")
	return(None)
    #if we got good image and need to save locally :
    if download:
	if not got_locally:   #only download if we didn't get file locally
    		if not os.path.isdir(download_directory):
        		os.makedirs(download_directory)
		if "://" in url_or_path_to_image_file_or_cv2_image_array:
	  		FILENAME = url_or_path_to_image_file_or_cv2_image_array.split('/')[-1].split('#')[0].split('?')[-1].split(':')[-1]
 			print('interim filename:'+str(FILENAME))
			FILENAME = os.path.join(download_directory,FILENAME)
		else:
 			FILENAME = os.path.join(download_directory,url_or_path_to_image_file_or_cv2_image_array)
			print('no //')
		if FILENAME.endswith('jpg')or FILENAME.endswith('jpeg') or FILENAME.endswith('.bmp') or FILENAME.endswith('tiff'):
			pass
		else:    #there's no 'normal' filename ending
			FILENAME=FILENAME+'.jpg' 
   		try:
			print('filename for local write:'+str(FILENAME))
        		write_status = imwrite(FILENAME,img_array)
    		except:
        		print('unexapected error in Utils calling imwrite')
#        		print('unexapected error in Utils calling urllib.urlretreive'+sys.exc_info()[0])
	

    return img_array

def count_human_bbs_in_doc(dict_of_images):
    n=0
    for entry in dict_of_images:
	if good_bb(entry):
		n=n+1   #got a good bb
    return(n)


def lookfor_next_unbounded_image(queryobject):
    n=0
    min_images_per_doc = 10
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
    if not 'human_bb' in dict:  
#	print('no human_bb key in dict')
	return(False)
    elif dict["human_bb"] is None:
#	print('human_bb is None')
	return(False)
    elif not(legal_bounding_box(dict["human_bb"])):
#	print('human bb is not big enough')
	return(False)
    else:
#	print('human bb ok:'+str(dict['human_bb']))
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
    training_collection_cursor = db.training.find()   #The db with multiple figs of same item
    doc = next(training_collection_cursor, None)
    resultDict = {}
    while doc is not None:
	if 'images' in doc:
		n = count_human_bbs_in_doc(doc['images'])
		print('number of good bbs:'+str(n))
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
