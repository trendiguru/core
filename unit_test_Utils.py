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


    def test_lookfor_and_insert(self):
        dict = Utils.test_lookfor_next()
        Utils.test_insert_bb(dict, [10, 20, 30, 40])

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



if __name__ == '__main__':
    unittest.main()


