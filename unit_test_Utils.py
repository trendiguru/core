import unittest

import pymongo
import numpy as np

import Utils

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    #    def testPass(self):
    #        return

    #    def testFail(self):
    #        self.failIf(True)

    #    def testError(self):
    #        raise RuntimeError('Test error!')

    def setUp(self):
        db = pymongo.MongoClient().mydb
        training_collection_cursor = db.training.find()  #The db with multiple figs of same item
        self.assertTrue(training_collection_cursor is not None)  #make sure training collection exists

    def test_lookfor_next_bounded_in_db_no_args(self):
    #answer should be a dictionary of info about bb or an error string if no bb found
        answer = Utils.lookfor_next_bounded_in_db()
        print('answer from lookfor_next_bounded_in_db_no_args:'+str(answer))
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, basestring))

    def test_lookfor_next_bounded_in_db(self):
    #answer should be a dictionary of info about bb or an error string if no bb found
        answer = Utils.lookfor_next_bounded_in_db(current_item=1, current_image=2,only_get_boxed_images=True)
        print('answer from lookfor_next_bounded_in_db:'+str(answer))
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, basestring))

    def test_lookfor_next_bounded_in_db_bad_args(self):
    #answer should be a dictionary of info about bb or an error string if no bb found
        answer = Utils.lookfor_next_bounded_in_db(current_item="1", current_image="2",only_get_boxed_images=True)
        print('answer from lookfor_next_bounded_in_db:'+str(answer))
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, basestring))

    def test_count_human_bbs_in_doc(self):
        db = pymongo.MongoClient().mydb
        training_collection_cursor = db.training.find()  # The db with multiple figs of same item
        doc = next(training_collection_cursor, None)
        dict_images = doc['images']
        n = Utils.count_human_bbs_in_doc(dict_images, skip_if_marked_to_skip=True)
        print('images:' + str(dict_images))
        print('n:' + str(n) + ' len(dict_images):' + str(len(dict_images)))
        self.assertTrue(n <= len(dict_images))

    def test_count_bbs(self):
        '''
        test counting how many good bb;s in doc
        '''

        db = pymongo.MongoClient().mydb
        training_collection_cursor = db.training.find()  # The db with multiple figs of same item
        doc = next(training_collection_cursor, None)
        resultDict = {}
        while doc is not None:
            if 'images' in doc:
                n = Utils.count_human_bbs_in_doc(doc['images'])
                print('number of good bbs:' + str(n))
            doc = next(training_collection_cursor, None)


    # test function for lookfor_next_unbounded_image
    def test_lookfor_next(self):
        db = pymongo.MongoClient().mydb
        training_collection_cursor = db.training.find()  # The db with multiple figs of same item
        doc = next(training_collection_cursor, None)
        resultDict = {}
        while doc is not None:
            url = None  # kill error
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
                prefix = 'none'  # kill error
                print("no unbounded image found for string:" + str(prefix) + " in current doc")
                # logging.debug("no unbounded image found for string:" + str(prefix) + " in current doc")
            doc = next(training_collection_cursor, None)
        return resultDict


    def test_bb_to_mask(self):
        img_array = np.array([[2, 0, 0, 2], [50, 0, 0, 50], [0, 23, 0, 25], [50, 0, 0, 50], [0, 23, 0, 25]])
        bb = [1, 0, 2, 2]
        mask = Utils.bb_to_mask(bb, img_array)
        print('img = ' + str(img_array))
        print('bb = ' + str(bb))
        print('mask = ' + str(mask))
        print('')
        self.assertTrue(mask.shape[0] == img_array.shape[0] and mask.shape[1] == img_array.shape[1])
        bb = [1, 0, 5, 6]
        mask = Utils.bb_to_mask(bb, img_array)
        print('img = ' + str(img_array))
        print('bb = ' + str(bb))
        print('mask = ' + str(mask))
        print('')
        self.assertTrue(mask.shape[0] == img_array.shape[0] and mask.shape[1] == img_array.shape[1])
        bb = [10, 20, 30, 40]
        mask = Utils.bb_to_mask(bb, img_array)
        print('img = ' + str(img_array))
        print('bb = ' + str(bb))
        print('mask = ' + str(mask))
        print('')
        self.assertTrue(mask.shape[0] == img_array.shape[0] and mask.shape[1] == img_array.shape[1])
        bb = [-10, 20, -30, 40]
        mask = Utils.bb_to_mask(bb, img_array)
        print('img = ' + str(img_array))
        print('bb = ' + str(bb))
        print('mask = ' + str(mask))
        self.assertTrue(mask.shape[0] == img_array.shape[0] and mask.shape[1] == img_array.shape[1])


if __name__ == '__main__':
    unittest.main()


