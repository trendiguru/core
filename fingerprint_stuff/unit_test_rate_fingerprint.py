__author__ = 'jeremy'
# import default
import unittest

import numpy as np

import rate_fp
import fingerprint_core
import NNSearch
import constants


class rate_fingerprint_test(unittest.TestCase):
    #examples of things to return
    #    def testPass(self):
    #        return

    #    def testFail(self):
    #        self.failIf(True)

    #    def testError(self):
    #        raise RuntimeError('Test error!')

    def setUp(self):
       # subprocess.call('sudo /home/jeremy/mongoloid.sh')
        pass


    def test_get_docs(self):
        n_items = 5
        report, docs = rate_fp.get_docs(n_items=n_items)
        l = len(docs)
        print('n:' + str(l) + ' docs:' + str(docs))
        print('report:' + str(report))
        self.assertTrue(l == n_items)


    def test_make_cross_comparison_sets(self):
        n_items = 4
        report, image_sets = rate_fp.get_docs(n_items=n_items)
        sets = rate_fp.make_cross_comparison_sets(image_sets)
        print(sets)
        self.assertTrue(len(sets) == n_items)

    def test_calculate_partial_cross_confusion_vector(self):
        n_items = 5
        report, image_sets = rate_fp.get_docs(n_items=n_items)
        report = rate_fp.calculate_partial_cross_confusion_vector(image_sets, fingerprint_function=fingerprint_core.fp,
                                                                  weights=np.ones(constants.fingerprint_length),
                                                                  distance_function=NNSearch.distance_1_k,
                                                                  distance_power=0.5, report=report)
        avg = report['average_weighted']
        print('cross item avg:' + str(avg))
        self.assertTrue(avg > 0 and avg < 100)  # this 100 is arbirary...

    def test_calculate_self_confusion_vector(self):
        n_items = 6
        report, image_sets = rate_fp.get_docs(n_items=n_items)
        report = rate_fp.calculate_self_confusion_vector(image_sets, fingerprint_function=fingerprint_core.fp,
                                                         weights=np.ones(constants.fingerprint_length),
                                                           distance_function=NNSearch.distance_1_k,
                                                         distance_power=0.5, report=report, use_visual_output1=False,
                                                         use_visual_output2=False)
        avg = report['average_weighted']
        print('self item avg:' + str(avg))
        self.assertTrue(avg > 0 and avg < 100)  # this 100 is arbirary...

    def test_analyze_fingerprint(self):
        n_items = 3
        tot_report = {}
        print('test the self_rate_fingerprint fucntion in rate_fingerprint')
        goodness, tot_report = rate_fp.analyze_fingerprint(fingerprint_function=fingerprint_core.fp,
                                                           weights=np.ones(constants.fingerprint_length),
                                                           distance_function=NNSearch.distance_1_k,
                                                           distance_power=0.5, n_docs=n_items)
        self.assertTrue(goodness > 0)
        print('after checking, report:' + str(tot_report))


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()


