__author__ = 'jeremy'
# import default
import unittest
import cProfile
import StringIO
import pstats
from scipy.spatial import distance as dist

import cv2
import numpy as np

import rate_fp
import fingerprint_core
import NNSearch
import constants


fingerprint_length = constants.fingerprint_length

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
                                                                  distance_power=0.5, report=report, parallelize=False)
        avg = report['average_weighted']
        print('cross item avg without parallelization:' + str(avg))
        self.assertTrue(avg > 0 and avg < 100)  # this 100 is arbirary...
        report = rate_fp.calculate_partial_cross_confusion_vector(image_sets, fingerprint_function=fingerprint_core.fp,
                                                                  weights=np.ones(constants.fingerprint_length),
                                                                  distance_function=NNSearch.distance_1_k,
                                                                  distance_power=0.5, report=report)
        avg2 = report['average_weighted']
        print('cross item avg with parallelization:' + str(avg2))
        self.assertTrue(avg > 0 and avg < 100)  # this 100 is arbirary...
        self.assertTrue(avg == avg2)  # this 100 is arbirary...

    def test_calculate_self_confusion_vector(self):
        n_items = 6
        report, image_sets = rate_fp.get_docs(n_items=n_items)
        report = rate_fp.calculate_self_confusion_vector(image_sets, fingerprint_function=fingerprint_core.fp,
                                                         weights=np.ones(constants.fingerprint_length),
                                                           distance_function=NNSearch.distance_1_k,
                                                         distance_power=0.5, report=report, parallelize=False)
        avg = report['average_weighted']
        print('self item avg without parallelization:' + str(avg))
        self.assertTrue(avg > 0 and avg < 100)  # this 100 is arbirary...
        report = rate_fp.calculate_self_confusion_vector(image_sets, fingerprint_function=fingerprint_core.fp,
                                                         weights=np.ones(constants.fingerprint_length),
                                                         distance_function=NNSearch.distance_1_k,
                                                         distance_power=0.5, report=report, parallelize=True)
        avg2 = report['average_weighted']
        print('self item avg with parallelization:' + str(avg2))
        self.assertTrue(avg == avg2)  # this 100 is arbirary...

    def test_analyze_fingerprint(self):
        n_items = 3
        tot_report = {}
        print('test the self_rate_fingerprint fucntion in rate_fingerprint')

        pr = cProfile.Profile()
        pr.enable()
        weights = np.ones(fingerprint_length)


        goodness, tot_report = rate_fp.analyze_fingerprint(fingerprint_function=fingerprint_core.fp,
                                                           weights=np.ones(constants.fingerprint_length),
                                                           distance_function=NNSearch.distance_1_k,
                                                           distance_power=0.5, n_docs=n_items)
        pr.disable()
        self.assertTrue(isinstance(goodness, float))
        print('after checking, report:' + str(tot_report))
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    def test_def_display_two_histograms(self):
        a = []
        b = []
        print('yo')
        for i in range(0, 300):
            a.append(np.random.normal(loc=2.0, scale=1))
            b.append(np.random.normal(loc=4.0, scale=0.5))
        # print(a,b)
        rate_fp.display_two_histograms(a, b)

    def test_compare_histograms(self):
        d1 = np.random.normal(loc=0.0, scale=1.0, size=20000)
        d2 = np.random.normal(loc=5.0, scale=1.0, size=20000)

        d1 = np.float32(d1)
        d2 = np.float32(d2)

        max1 = max(d1)
        max2 = max(d2)
        maxboth = max(max1, max2)
        minboth = min(min(d1), min(d2))

        hist1, binsout = np.histogram(d1, range=(minboth, maxboth), bins=40)
        hist1 = np.float32(hist1)
        hist1 = cv2.normalize(hist1).flatten()

        hist2, binsout = np.histogram(d2, range=(minboth, maxboth), bins=40)
        hist2 = np.float32(hist2)
        hist2 = cv2.normalize(hist2).flatten()

        rate_fp.display_two_histograms(hist1, hist2, binsout)

        print('euclidean dist:' + str(dist.euclidean(hist1, hist2)))
        print('cityblock dist:' + str(dist.cityblock(hist1, hist2)))
        print('chebyshev dist:' + str(dist.chebyshev(hist1, hist2)))
        print('correlation:' + str(
            cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)))
        print('chisqr:' + str(
            cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CHISQR)))
        print('intersection:' + str(
            cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_INTERSECT)))
        print('bhatta:' + str(
            cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_BHATTACHARYYA)))

        # results = { #"Correlation":cv2.compareHist(hist1,hist2,cv2.cv.CV_COMP_CORREL),
        # "Chi-Squared":cv2.compareHist(hist1,hist2,cv2.cv.CV_COMP_CHISQR),
        # "Intersection":cv2.compareHist(hist1,hist2,cv2.cv.CV_COMP_INTERSECT),
        # "Bhattacharyya":cv2.compareHist(hist1,hist2, cv2.cv.CV_COMP_BHATTACHARYYA),


def display_two_histograms(same_distances, different_distances, name=None):


    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()


