import unittest
import pymongo

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
        assert training_collection_cursor is not None  #make sure training collection exists

    def test_lookfor_next_bounded_in_db(self):
    #answer should be a dictionary of info about bb or an error string if no bb found
        answer = Utils.lookfor_next_bounded_in_db()
        print('answer from lookfor_next_bounded_in_db:'+str(answer))
        self.assertTrue(isinstance(answer, dict) or isinstance(answer, basestring))

if __name__ == '__main__':
    unittest.main()


