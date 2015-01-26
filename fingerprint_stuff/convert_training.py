__author__ = 'jr'
import pymongo
import logging	
import json

db = pymongo.MongoClient().mydb
#    product_collection= db.products
training_collection= db.training

records = training_collection.find()

for doc in records:
	image_array=[]
#	print('before:'+ json.loads(str(doc)) )
	print('before:'+ str(doc) )
	training_collection.update({'_id':doc['_id']}, {'$unset':{'array_of_images':''}},upsert=True)
	training_collection.update({'_id':doc['_id']}, {'$unset':{'images':''}},upsert=True)
	for key in doc.keys():
		if 'image' in key or 'Image' in key:
			if key != 'images' and key != 'array_of_images':  #this is to prevent growing lists since the word image is contained in array_OF_images

				url=doc[key]
				image_array.append({'url':doc[key],'human_bb':None,'old_name':key})
			else:
				print('currently reservedkey')
	training_collection.update({'_id':doc['_id']}, {'$set':{'images':image_array}},upsert=True)
	#doc.update({'_id':mongo_id}, {"$set": post}, upsert=False)
    	print('id:'+str(doc['_id']))	
	changedRecord = training_collection.find_one({'_id':doc['_id']})
	print()
        print('after:'+str(changedRecord))
	a = raw_input()
#remove the now-duplicate info later so i don't risk losing info
#	doc.update({'images':images})


