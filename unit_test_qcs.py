__author__ = 'jeremy'
import unittest

import pymongo
import bson

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


    # TODO  - change to test other code that's doing this
    # def test_determine_final_bb(self):
    # bb1 = [10, 20, 100, 100]
    # bb2 = [10, 20, 1, 2]
    # bb3 = [10, 20, 102, 104]
    #        bb4 = [10, 20, 20, 30]
    #        bb_list = [bb1, bb2, bb3, bb4]
    #        final_bb = qcs.determine_final_bb(bb_list)
    #        print('final bb:' + str(final_bb))
    #        self.assertTrue(final_bb[0] == 10)
    #        self.assertTrue(final_bb[1] == 20)
    #        self.assertTrue(final_bb[2] == 101)
    #        self.assertTrue(final_bb[3] == 102)

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
            self.assertTrue(n_votes_per_picture >= 1)

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

    def test_get_item_by_id(self):
        print('testing get_item_by_id')
        image, item, person = qcs.get_item_by_id(bson.ObjectId('55b8a8b61f8c825656f14b40'))
        print('item:' + str(item))
        print('image:' + str(item))
        print('person:' + str(item))


    def test_set_voting_stage(self):
        stage = 0
        qcs.set_voting_stage(stage, bson.ObjectId('55b8a8b61f8c825656f14b40'))
        actual_stage = qcs.get_voting_stage(bson.ObjectId('55b8a8b61f8c825656f14b40'))
        print('voting_stage=' + str(actual_stage))
        self.assertTrue(stage == actual_stage)

    def test_get_voting_stage(self):
        stage = qcs.get_voting_stage(bson.ObjectId('55b8a8b61f8c825656f14b40'))
        print('voting_stage=' + str(stage))
        self.assertTrue(isinstance(stage, int))

        # vote count is currently wrong
    def test_combine_votes(self):
        votes_list = [[4, 5, 6], [2, 10], ['not relevant', 'not relevant'], ['not relevant', 3, 10]]
        for votes in votes_list:
            res = qcs.combine_votes(votes)
            print('votes:' + str(votes) + ' res:' + str(res))


    def test_add_results(self):
        extant_items = ['trump', 'clinton', 'libertarian guy', 'wildcard']
        extant_votes = [[4, 5, 6], [2, 10], ['not relevant', 'not relevant'], ['not relevant', 3, 10]]
        new_items = ['trump', 'libertarian guy', 'clinton', 'al gore', 'wierd al yankovic']
        new_votes = [7, 1, 'not relevant', 3, 6]

        print('extant_items:' + str(extant_items))
        print('extant_votes:' + str(extant_votes))
        print('new items:' + str(new_items))
        print('new votes:' + str(new_votes))
        tot_votes, combined_items, combined_votes = qcs.add_results(extant_items, extant_votes, new_items, new_votes)
        print('tot_votes:' + str(tot_votes))
        print('combined items:' + str(combined_items))
        print('combined votes:' + str(combined_votes))

    # TODO add asserts here for right answers

    def test_order_results(self):
        items = ['trump', 'clinton', 'libertarian guy', 'wildcard']
        votes = [[4, 5, 6], [2, 10], ['not relevant', 'not relevant'], ['not relevant', 3, 10]]
        print('initial items:' + str(items))
        print('initial votes:' + str(votes))
        sorted_items, sorted_votes = qcs.order_results(items, votes)
        print('ordered items:' + str(sorted_items))
        print('ordered votes:' + str(sorted_votes))
        self.assertTrue(sorted_items == ['wildcard', 'trump', 'clinton', 'libertarian guy'])
        self.assertTrue(sorted_votes == [4.0, 5.0, 6.0, 'not relevant'])


    def test_from_qc_get_votes(self):
        item_id = bson.ObjectId('55b8a8b61f8c825656f14b40')
        similar_items = ['a', 'b']
        votes = [4, 3]
        voting_stage = 1
        qcs.from_qc_get_votes(item_id, similar_items, votes, voting_stage)

    if __name__ == '__main__':
        unittest.main()

