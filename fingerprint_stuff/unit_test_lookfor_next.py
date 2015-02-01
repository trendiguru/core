import cv2
import urllib
import pymongo
import os
import urlparse
#import default
import find_similar_mongo
import unittest
import imp
import sys

#myfile,mypathname,mydescription = imp.find_module('default.py','/home/www-data/web2py/applications/REST4/controllers/')
#imp.load_module('default',myfile, mypathname, mydescription)

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
#a=True
    	assert(self.training_collection_cursor)  #make sure a exists
#        self.parser = LammpsDataFileParser(handler=self.handler)
#        self.filename = os.path.join(self.temp_dir, "test_data.txt")

#    def tear_down(self):
#        shutil.rmtree(self.temp_dir)

    def test_lookfor_next(self):
	print('path='+str(sys.path))
    	fs = find_similar_mongo  #yeah yeah use mongo
    
    #get image and bounding box if it exists
        print('reached GET function in default.py: vars are '+str(vars))
        resultDict = []
        #if "bb" in vars:   #not using this, i assume all these gets are for next image without bounding box
        resultDict = {}  #return empty dict if no results found
        prefixes = ['Main Image URL angle ', 'Style Gallery Image ']

        doc = next(self.training_collection_cursor, None)
        while doc is not None:
            #training docs contains lots of different images (URLs) of the same clothing item
            print('doc:'+str(doc))
            for prefix in prefixes:
                url = fs.lookfor_next_unbounded_image(doc, prefix)
                if url:
                    resultDict["image url"] = url
                    resultDict["id"] = str(doc['_id'])
		    print('result:'+str(resultDict))   
 		    assert(resultDict)
                    return resultDict
                else:
                    logging.debug("no unbounded image found for string:" + str(prefix))
    	assert(resultDict)


#		strN='Style Gallery Image '+str(n)
#		print('looking for string:'+str(strN))




if __name__ == '__main__':
    unittest.main()


