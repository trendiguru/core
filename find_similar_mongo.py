__author__ = 'liorsabag'

import pymongo
import fingerPrint2 as fp
from NNSearch import findNNs
#from fingerprint_db_params_mongo import get_all_subcategories
import logging

def get_all_subcategories(category_collection, category_id):
    subcategories = []

    def get_subcategories(c_id):
        subcategories.append(c_id)
        curr_cat = category_collection.find_one({"id": c_id})
        if "childrenIds" in curr_cat.keys():
            for childId in curr_cat["childrenIds"]:
                get_subcategories(childId)

    get_subcategories(category_id)
    return subcategories


def find_with_bb_and_keyword(imageURL, bb, category_id, number_of_results=10):
    db = pymongo.MongoClient().mydb
    product_collection = db.products

    subcategory_id_list = get_all_subcategories(db.categories, category_id)

    # get all items in the subcategory/keyword
    query = product_collection.find({"$and": [{"categories": {"$elemMatch": {"id": {"$in": subcategory_id_list}}}},
                                             {"fingerprint": {"$exists": 1}}]},
                                    {"_id": 0, "id": 1, "categories": 1, "fingerprint": 1, "image": 1,
                                     "clickUrl": 1, "price": 1, "brand": 1})
        # {"$and": [
        #     {"categories": {"$elemMatch": {"id": {"$in": subcategory_id_list}}}},
        #     #{"categories": {"$elemMatch": {"name": keyword}}},
        #     {"fingerprint": {"$exists": 1}}]},
        # {  # What properties to select
        #    "_id": 0,
        #    "id": 1,
        #    "categories": 1,
        #    "fingerprint": 1,
        #    "image": 1,
        #    "clickUrl": 1
        # })

    db_fingerprint_list = []
    for row in query:
        fp_dict = {}
        fp_dict["id"] = row["id"]
        fp_dict["clothingClass"] = category_id
        fp_dict["fingerPrintVector"] = row["fingerprint"]
        fp_dict["imageURL"] = row["image"]["sizes"]["Large"]["url"]
        fp_dict["buyURL"] = row["clickUrl"]
        db_fingerprint_list.append(fp_dict)

    #Fingerprint the bounded area
    fgpt = fp.fp(imageURL, bb)
    target_dict = {"clothingClass": category_id, "fingerPrintVector": fgpt}

    closest_matches = findNNs(target_dict, db_fingerprint_list, number_of_results)
    return fgpt.tolist(), closest_matches

#this is for the training collection, where there's a set of images from different angles in each record
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

def find_next_unbounding_boxed_item(query_result):
    db = pymongo.MongoClient().mydb
#    product_collection= db.products
    training_collection= db.training

    subcategory_id_list = get_all_subcategories(db.categories, category_id)
    query_by_keyword = False
    
    #get all items in the subcategory/keyword  
    if query_by_keyword:
    	query = training_collection.find({"$and": [{"categories": {"$elemMatch": {"id": {"$in": subcategory_id_list}}}},
                                             {"fingerprint": {"$exists": 1}}]},
#                                    {"_id": 0, "id": 1, "categories": 1, "fingerprint": 1, "image": 1, "clickUrl": 1})
                                    {"_id": 0, "id": 1, "categories": 1, "image": 1, "clickUrl": 1}) #dont care if there is fingerprint or not

    #get first item that doesn't have a bb
    else:  
	query = training_collection.find()
	while training_collection.hasnext:
	#raw_input('Enter a file name: ...')
		print('total record:'+str(query))
    		strN='Main Image URL angle '
    		lookfor_consecutive_strings_in_query(b,strN)
    		strN='Style Gallery Image '
    		lookfor_consecutive_strings_in_query(b,strN)
		b=a.next()

	

         # {"$and": [
        #     {"categories": {"$elemMatch": {"id": {"$in": subcategory_id_list}}}},
        #     #{"categories": {"$elemMatch": {"name": keyword}}},
        #     {"fingerprint": {"$exists": 1}}]},
        # {  # What properties to select
        #    "_id": 0,
        #    "id": 1,
        #    "categories": 1,
        #    "fingerprint": 1,
        #    "image": 1,
        #    "clickUrl": 1
        # })

#probably unecessary stuff 
    db_fingerprint_list = []
    for row in query:
        fp_dict = {}
        fp_dict["id"] = row["id"]
        fp_dict["clothingClass"] = category_id
        fp_dict["fingerPrintVector"] = row["fingerprint"]
        fp_dict["imageURL"] = row["image"]["sizes"]["Large"]["url"]
        fp_dict["buyURL"] = row["clickUrl"]
        db_fingerprint_list.append(fp_dict)

    #Fingerprint the bounded area
    fgpt = fp.fp(imageURL, bb)
    target_dict = {"clothingClass": category_id, "fingerPrintVector": fgpt}

    closest_matches = findNNs(target_dict, db_fingerprint_list, number_of_results)
    return fgpt.tolist(), closest_matches
