__author__ = 'jeremy'
import cv2
import urllib
import pymongo
import os
import urlparse
# import default
import Utils
import unittest
import imp
import sys
import rate_fingerprint
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
        pass

    def test_rate_fingerprint(self):
        print('test the self_rate_fingerprint fucntion in rate_fingerprint')
        rating = rate_fingerprint.self_rate_fingerprint(fingerprint_function=fingerprint_core.fp,
                              weights=np.ones(constants.fingerprint_length), distance_function=NNSearch.distance_1_k,
                              distance_power=1.5)
        unittest.assertTrue(rating>0)

    def test_calculate_self_confusion_vector(self):
        confusion_vector = calculate_self_confusion_vector(fingerprint_function=fingerprint_function, weights=weights,
                                                           distance_function=distance_function,
                                                           distance_power=distance_power)

        unittest.assertTrue(isinstance(confusion_vector,np.ndarray))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()


