__author__ = 'jr'

import pymongo
import logging

def add_bounding_box(collection,id,image_description_string,chosen_bounding_box)
    '''
    given a bb and db record, add bb to record
    '''
    print('bb:'+str(chosen_bounding_box))
    logging.debug('adding for bb:'+str(chosen_bounding_box))	
    collection.update({"id": doc["id"]},
    	{"$set": {"human_bounding_box": np.array(chosen_bounding_box).tolist()}   })
      
#    	"fp_date": time.strftime('%Y-%m-%d %H:%M:%S'),

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
		if not bbN in queryobject:  # got a pic without a bb
			urlN=queryobject[strN]
			got_unbounded_image = True
		else:
			print('image from string:'+strN+' :is bounded')
	else:
		print('didn\'t find expected string in training db')
		logging.debug('didn\'t find expected string in training db')
		break	
    return(urlN)
# maybe return(urlN,n) at some point

def keyword_query(query_result):
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
