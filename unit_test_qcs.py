__author__ = 'jeremy'
import unittest

import pymongo

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
        db = pymongo.MongoClient().mydb

        images_entry = db.images.find_one()  # The db with multiple figs of same item
        self.assertTrue(images_entry is not None)  # make sure images collection exists

    def test_from_image_URL_to_task1(self):
        # url = 'http://static1.1.sqspcdn.com/static/f/163930/1599100/1211882974117/justinpolkey2.jpg'
        url = 'http://i.huffpost.com/gen/1321037/thumbs/o-KATE-LANPHEAR-570.jpg'
        retval = qcs.from_image_url_to_task1(url)
        print(retval)

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

    def test_at_least_one_vote_per_image(self):
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

    def test_get_voting_stage(self):
        db = pymongo.MongoClient().mydb
        images_entry = db.images.find_one()  # The db with multiple figs of same item

        if 'people' in images_entry:
            first_person = images_entry['people'][0]
            if 'items' in first_person:
                first_item = first_person['items'][0]
                if 'item_id' in first_item:
                    item_id = first_item['item_id']
                    n = qcs.get_voting_stage(item_id)
                    print('entry ' + str(images_entry))
                    print('person ' + str(first_person))
                    print('item ' + str(first_item))
                    print('item_id ' + str(item_id))
                    print('voting stage ' + str(n))
                else:
                    print('no item_id in item ' + str(first_item))
            else:
                print('no items in person ' + str(first_person))
        else:
            print('no people in images_entry ' + str(images_entry))



if __name__ == '__main__':
    unittest.main()


