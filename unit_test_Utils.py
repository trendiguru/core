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


if __name__ == '__main__':
    unittest.main()


