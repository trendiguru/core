import cv2
import urllib
import pymongo
import os
import urlparse
#import default
import Utils
import unittest
import imp
import sys


class OutcomesTest(unittest.TestCase):

#examples of things to return
#    def testPass(self):
#        return

#    def testFail(self):
#        self.failIf(True)

#    def testError(self):
#        raise RuntimeError('Test error!')

    def setUp(self):
	import pymongo
	db=pymongo.MongoClient().mydb
	self.training_collection_cursor = db.training.find()   #The db with multiple figs of same item
	import find_similar_mongo
    	assert(self.training_collection_cursor)  #make sure training collection exists

#    def tear_down(self):
#        shutil.rmtree(self.temp_dir)

    def test_lookfor_next(self):
	print('path='+str(sys.path))
    
    #get image and bounding box if it exists
        print('reached GET function in default.py: vars are '+str(vars))
        resultDict = {}  #return empty dict if no results found
        prefixes = ['Main Image URL angle ', 'Style Gallery Image ']

        doc = next(self.training_collection_cursor, None)
        while doc is not None:
            print('doc:'+str(doc))
            for prefix in prefixes:
                url = Utils.lookfor_next_unbounded_image(doc, prefix)
                if url:
                    resultDict["image url"] = url
                    resultDict["id"] = str(doc['_id'])
		    print()
		    print('result:'+str(resultDict))   
 		    self.assertTrue(resultDict)
		    imagelist = doc['images']
		    for image in imagelist:
			if image['url'] == url:
				print('image record is:'+str(image))
				self.assertTrue(image['human_bb'] == None)
				print('human bb absent has been checked')
			    	self.assertTrue(resultDict)
	   	                return resultDict
		    self.assertTrue(False,msg='did not find the url in the document')
		    print('imageslist:'+str(imagelist))		
# add assert that bounding box is empty , along lines of doc[images][url]['bb'] is not null
                else:
                    logging.debug("no unbounded image found for string:" + str(prefix))
		    self.assertTrue(False,msg='did not receive url from lookfor_next_unbounded_image')

#		strN='Style Gallery Image '+str(n)
#		print('looking for string:'+str(strN))




if __name__ == '__main__':
    unittest.main()


