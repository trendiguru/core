import cv2
import urllib
import pymongo
import os
import urlparse

import find_similar_mongo

def assure_path_exists(path):
#    print('assuring existence of path:'+str(path))
    dir=os.path.dirname(path)
    if not os.path.exists(path):
    	os.makedirs(path)

def read_img(path_to_image_file):
    localdir='images'
    REMOTE_FILE = False
    item_found = False
    localpath=localdir  #make a separate directory for each item
    assure_path_exists(localpath)

    if path_to_image_file is not None:
	PATH_TO_FILE=path_to_image_file
    #we want to be able to read URL as well as local file path
    	if "://" in path_to_image_file:
#        	FILENAME = path_to_image_file.split('/')[-1].split('#')[0].split('?')[0]
#		FILENAME = os.path.join(objID,FILENAME) 
#		FILENAME = urllib.url2pathname(path_to_image_file)
		pr = urlparse.urlparse(path_to_image_file)
       	 	FILENAME = pr.path[1:]
		FILENAME = FILENAME.replace('/','_')	
		FILENAME = os.path.join(localpath,FILENAME)
		print('url:'+path_to_image_file+' localPath:'+FILENAME)
		res = urllib.urlretrieve(PATH_TO_FILE, FILENAME)

    use_visual_output = False
    if use_visual_output:
    	img = cv2.imread(FILENAME)
    	cv2.imshow('image',img)
    	cv2.waitKey(0) 	
    	cv2.destroyAllWindows()






#######################
# MAIN
#######################

localdir='images'
db=pymongo.MongoClient().mydb
#a=True
a=db.training.find()
print(a)
i=0
while db.training.hasnext:
	#raw_input('Enter a file name: ...')
	if i==0:
		b=a.next()
	else:
		b=a.next()
	print('total record:'+str(b))
	urls_remain=True

	strN='Main Image URL angle '
	url=find_similar_mongo.lookfor_next_unbounded_image(b,strN)
	if url:
		read_img(url)

	strN='Style Gallery Image '
	url=find_similar_mongo.lookfor_next_unbounded_image(b,strN)
	if url:
		read_img(url)
	i=i+1

#		strN='Style Gallery Image '+str(n)
#		print('looking for string:'+str(strN))


