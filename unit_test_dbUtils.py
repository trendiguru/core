__author__ = 'jeremy'
import unittest

import pymongo

import dbUtils


class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return

    # def testFail(self):
    # self.failIf(True)

    # def testError(self):
    # raise RuntimeError('Test error!')

    def setUp(self):
        db = pymongo.MongoClient().mydb
        training_collection_cursor = db.training.find()  #The db with multiple figs of same item
        self.assertTrue(training_collection_cursor is not None)  #make sure training collection exists


    def test_lookfor_next_unbounded_feature_from_db_category(self):
        category_id = 'v-neck-sweaters'
        word = 'neck'
        ftype = 'byWordInDescription'
        # answer = dbUtils.lookfor_next_unbounded_feature_from_db_category(current_item=0,skip_if_marked_to_skip=True,which_to_show='showUnboxed',filter_type='byWordInDescription',category_id=category_id,word_in_description=None,db=None)
        answer = dbUtils.lookfor_next_unbounded_feature_from_db_category(item_number=0, skip_if_marked_to_skip=True,
                                                                         which_to_show='showUnboxed', filter_type=ftype,
                                                                         category_id=category_id,
                                                                         word_in_description=word)
        print('answer from lookfor_next_unbounded_feature_from_db:' + str(answer))
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, basestring))
        #


if __name__ == '__main__':
    unittest.main()


