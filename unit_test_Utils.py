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
import pymongo


class OutcomesTest(unittest.TestCase):

#examples of things to return
#    def testPass(self):
#        return

#    def testFail(self):
#        self.failIf(True)

#    def testError(self):
#        raise RuntimeError('Test error!')

    def setUp(self):
        db=pymongo.MongoClient().mydb
        self.training_collection_cursor = db.training.find()   #The db with multiple figs of same item
        assert(self.training_collection_cursor)  #make sure training collection exists


    def test_lookfor_next(self):
		while doc is not None:
            #training docs contains lots of different images (URLs) of the same clothing item
            logger.debug(str(doc))
#           print('doc:'+str(doc))
            url,bb,skip = Utils.lookfor_next_unbounded_image(doc)
            if url:
                resultDict["url"] = url
                resultDict["bb"] = bb
                resultDict["skip"] = skip
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
                doc = next(training_collection_cursor, None)
                logger.debug("no bounded image found in current doc, trying next")
        logger.debug("no bounded image found in collection")
        return None

        doc = next(self.training_collection_cursor, None)




if __name__ == '__main__':
    unittest.main()


