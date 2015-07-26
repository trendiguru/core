__author__ = 'jeremy'
import unittest

import qcs
import constants

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')

    def setUp(self):
        pass

        # db = pymongo.MongoClient().mydb

    # images_collection_cursor = db.images.find()  #The db with multiple figs of same item
    # self.assertTrue(images_collection_cursor is not None)  #make sure images collection exists

    def test_determine_final_bb(self):
        bb1 = [10, 20, 100, 100]
        bb2 = [10, 20, 1, 2]
        bb3 = [10, 20, 102, 104]
        bb4 = [10, 20, 20, 30]
        bb_list = [bb1, bb2, bb3, bb4]
        final_bb = qcs.determine_final_bb(bb_list)
        print('final bb:' + str(final_bb))
        self.assertTrue(final_bb[0] == 10)
        self.assertTrue(final_bb[1] == 20)
        self.assertTrue(final_bb[2] == 101)
        self.assertTrue(final_bb[3] == 102)

    def test_at_least_one_vote(self):
        # N_bb_votes_required = 2
        # N_category_votes_required = 2
        self.assertTrue(len(constants.N_workers) == len(constants.N_pics_per_worker))
        self.assertTrue(len(constants.N_workers) == len(constants.N_top_results_to_show))
        for i in range(0, len(constants.N_workers)):
            print('stage ' + str(i) + ':N results to show:' + str(constants.N_top_results_to_show[i]))
            print('N workers:' + str(constants.N_workers[i]))
            print('N pics per worker:' + str(constants.N_pics_per_worker[i]))
            n_votes = constants.N_workers[i] * constants.N_pics_per_worker[i]
            n_votes_per_picture = n_votes / constants.N_top_results_to_show[i]
            print('n_votes:' + str(n_votes) + ', votes per picture:' + str(n_votes_per_picture) + '\n')
            self.assertTrue(n_votes_per_picture > 1)

if __name__ == '__main__':
    unittest.main()


