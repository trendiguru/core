__author__ = 'liorsabag'
import sys
from Utils import get_cv2_img_array
import fingerprint_core as fp
import classify_core as classify
from bson import datetime
import logging
import pymongo
import numpy as np
import urllib
import os
import cv2
import time

def get_all_subcategories(category_collection, category_id):
    subcategories = []

    def get_subcategories(c_id):
        subcategories.append(c_id)
        curr_cat = category_collection.find_one({"id": c_id})
        if "childrenIds" in curr_cat.keys():
            for childId in curr_cat["childrenIds"]:
                get_subcategories(childId)

    get_subcategories(category_id)
    print(subcategories)
    return subcategories

def get_size_from_url(url):
    item_found = False
    temp_dir = 'temp'
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    #put images in local directory
    FILENAME = url.split('/')[-1].split('#')[0].split('?')[0]
    FILENAME = os.path.join(temp_dir,FILENAME)
    try:
	res = urllib.urlretrieve(url, FILENAME)
    except:
    	print('unexapected error:'+sys.exc_info()[0])
    #pdb.set_trace()
    #main prog starts here
    img = cv2.imread(FILENAME)
    h = img.shape[0]
    w = img.shape[1]
    os.remove(FILENAME)
    return(w,h)

def fingerprint_the_unfingerprinted():
    """
    fingerprint everything in db - should not overwrite currently-written stuff
    """    
    db = pymongo.MongoClient().mydb
    print('fingerprinting unfingerprinted items')
    query = db.products.find().batch_size(100) # batch_size required because cursor timed out without it
    total_items = query.count()
    alpha = 0.01
    i = 1
    n_human_boxed = 0
    n_existing_boxes = 0
    n_unbounded_images = 0
    previous_tick = time.time()
    start_time = time.time()
    rate = 0
    for doc in query:
    	tick = time.time()
    	dt = tick - previous_tick
    	Dt = tick - start_time
    	rate = alpha/dt + (1.0-alpha)*rate
    	Dt_expected=float(total_items)/rate
    	remaining_time = float(total_items-i)/rate
    	previous_tick = tick
        print "Starting %d of %d (%.5f percent), time %.3f of %.3f (%.5f percent), rate:%.3f images/s, dt:%6f" % (i,total_items,100.0*float(i)/float(total_items),Dt,Dt_expected,100.0*float(Dt)/float(Dt_expected),rate,dt)
        image_url = doc["image"]["sizes"]["XLarge"]["url"]
        print "Image URL: {0}".format(image_url)
        # if there is a valid human BB, skip it
        if "human_bb" in doc.keys() and doc["human_bb"] != [0, 0, 0, 0] and doc["human_bb"] is not None:
            chosen_bounding_box = doc["human_bb"]
    	    print('human bounding_box:'+str(chosen_bounding_box))
            logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
    	    n_human_boxed += 1
        #otherwise if there is a valid automatically generated bb skip it
        elif "bounding_box" in doc.keys() and doc["bounding_box"] != [0,0,0,0] and doc["bounding_box"] is not None:
    	    chosen_bounding_box = doc["bounding_box"]
	    print('existing bounding_box:'+str(chosen_bounding_box))
            logging.debug("classifier bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
    	    if chosen_bounding_box[0] == 0 and chosen_bounding_box[1] == 0:
		n_unbounded_images += 1
    	    else:
    		n_existing_boxes += 1
        else:
    	    w,h = get_size_from_url(image_url)
    	    chosen_bounding_box = [0,0,w,h]
            try:
            	fingerprint = fp.fp(get_cv2_img_array(image_url), chosen_bounding_box)
            	db.products.update({"id": doc["id"]},
                                      {"$set": {"fingerprint": fingerprint.tolist(),
                                                "fp_date": datetime.datetime.now(),
                                                "bounding_box": np.array(chosen_bounding_box).tolist()}
                                      })
 	   	print('full image bounding_box:'+str(chosen_bounding_box))
    		n_unbounded_images += 1
            except Exception as e:
            	logging.warning("Exception caught while fingerprinting, skipping: {0}".format(e))
        i += 1
    	print('auto-boxed images:'+str(n_existing_boxes)+' human-boxed images:'+str(n_human_boxed)+' unboxed images:'+str(n_unbounded_images))
#    	s = raw_input('hit return for next')

    return(n_existing_boxes,n_human_boxed,n_unbounded_images)

def main():
    """
    example: python fingerprint_db_params_mongo.py mens-shirts shirtClassifier.xml true
    to run on entire database and fingerprint anything not that doesnt already have a fingerprint, do:
    python fingerprint_db_params_mongo.py undone
    """
# logging.basicConfig(filename='fingerprint_db.log', level=logging.DEBUG)
    first_run = False
    print(sys.argv)
    if len(sys.argv) == 2:
        category_name = sys.argv[1]
    	if category_name in ['undone','Undone','pending']: #fingerprint everything that doesn't currently have a fingerprint
    		return(fingerprint_the_unfingerprinted())
    	else:
    		return({'error':'dont know what the fingerprint, maybe missing parameters?'})
    if len(sys.argv) >= 3:
        category_name = sys.argv[1]
        classifier_xml = sys.argv[2]
        if len(sys.argv) == 4 and sys.argv[3].lower() == "true":
            first_run = True
    else:
        print("Missing parameters. Example use: python fingerprint_db_params_mongo.py mens-shirts shirtClassifier.xml true")
    	return({'error':'missing parameters'})	

    db = pymongo.MongoClient().mydb
    query_doc = {}
    #during first run we should record date
    if first_run is True:
        db.globals.update({"_id": "FP_DATE"}, {"$set": {category_name: datetime.datetime.now()}})
        query_doc = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}}

    #else find all docs with date earlier than date recorded in first_run
    else:
        fp_start_date = db.globals.find_one({"_id": "FP_DATE"})[category_name]
        query_doc = {"$and": [
            {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_name)}}}},
            {"fp_date": {"$lt": fp_start_date}}]}

    query = db.products.find(query_doc).batch_size(100) # batch_size required because cursor timed out without it

    total_items = query.count()

    i = 1
    # boxes_found = 0

    for doc in query:
        print "Starting {i} of {total}...".format(i=i, total=total_items)
        image_url = doc["image"]["sizes"]["XLarge"]["url"]
        print "Image URL: {0}".format(image_url)

        # if there is a valid human BB, use it
        if "human_bb" in doc.keys() and doc["human_bb"] != [0, 0, 0, 0] and doc["human_bb"] is not None:
            chosen_bounding_box = doc["human_bb"]
            logging.debug("Human bb found: {bb} for item: {id}".format(bb=chosen_bounding_box, id=doc["id"]))
        #otherwise use the largest of possibly many classifier bb's
        else:
            try:
                bounding_box_list = classify.classify_image_with_classifiers(image_url, classifier_xml)[classifier_xml]
            except KeyError:
                logging.warning("Could not classify with {0}".format(classifier_xml))
                bounding_box_list = []

            max_bb_area = 0
            chosen_bounding_box = None
            for possible_bb in bounding_box_list:
                if possible_bb[2] * possible_bb[3] > max_bb_area:
                    chosen_bounding_box = possible_bb
                    max_bb_area = possible_bb[2] * possible_bb[3]
            if chosen_bounding_box is None:
                chosen_bounding_box = [0, 0, 0, 0]
                logging.warning("No Bounding Box found, using [0,0,0,0]. "
                                "Document id: {0}, BB_list: {1}".format(doc["id"], str(bounding_box_list)))

        try:
            fingerprint = fp.fp(get_cv2_img_array(image_url), chosen_bounding_box)

            db.products.update({"id": doc["id"]},
                                      {"$set": {"fingerprint": fingerprint.tolist(),
                                                "fp_date": datetime.datetime.now(),
                                                "bounding_box": np.array(chosen_bounding_box).tolist()}
                                      })
            i += 1

        except Exception as e:
            logging.warning("Exception caught while fingerprinting, skipping: {0}".format(e))


if __name__ == "__main__":
    main()

"""
def print_children(category_id, hierarchy_string):
    curr_cat = category_collection.find_one({"id": category_id})
    print hierarchy_string + curr_cat["name"]
    if "childrenIds" in curr_cat.keys():
        for childId in curr_cat["childrenIds"]:
            printChildren(childId, " - - " + hierarchy_string)
"""


