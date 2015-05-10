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


    def test_get_next_unbounded_feature_from_db_category(self):
        category_id = 'v-neck-sweaters'
        answer = dbUtils.lookfor_next_unbounded_feature_from_db_category(category_id=category_id)
        print('answer from lookfor_next_bounded_in_db_no_args:' + str(answer))
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, basestring))
        #


if __name__ == '__main__':
    unittest.main()


