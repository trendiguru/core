#import default
import unittest
import sys

import Utils
from .constants import db

class OutcomesTest(unittest.TestCase):


    def setUp(self):
        self.training_collection_cursor = db.training.find()  # The db with multiple figs of same item
        assert (self.training_collection_cursor)  # make sure training collection exists

#    def tear_down(self):
#        shutil.rmtree(self.temp_dir)

    def test_lookfor_next(self):
        print('path=' + str(sys.path))
    
    #get image and bounding box if it exists
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
            print('result:' + str(resultDict))
            self.assertTrue(resultDict)
            imagelist = doc['images']
            for image in imagelist:
                if image['url'] == url:
                    print('image record is:' + str(image))
                    self.assertTrue(image['human_bb'] == None)
                    print('human bb absent has been checked')
                    self.assertTrue(resultDict)
                    return resultDict
# add assert that bounding box is empty , along lines of doc[images][url]['bb'] is not null
                else:
                    self.assertTrue(False, msg='did not receive url from lookfor_next_unbounded_image')
                    self.assertTrue(False, msg='did not find the url in the document')
                    print('imageslist:' + str(imagelist))

# strN='Style Gallery Image '+str(n)1111
#		print('looking for string:'+str(strN))




if __name__ == '__main__':
    unittest.main()


