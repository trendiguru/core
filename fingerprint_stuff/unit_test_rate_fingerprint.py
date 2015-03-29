__author__ = 'jeremy'
import cv2
import urllib
import pymongo
import os
import urlparse
# import default
import unittest
import imp
import sys
import numpy as np
import subprocess

import rate_fingerprint
import fingerprint_core
import Utils
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

    def test_rate_fingerprint(self):
        print('test the self_rate_fingerprint fucntion in rate_fingerprint')
        rating = rate_fingerprint.self_rate_fingerprint(fingerprint_function=fingerprint_core.fp,
                              weights=np.ones(constants.fingerprint_length), distance_function=NNSearch.distance_1_k,
                              distance_power=0.5)
        unittest.assertTrue(rating>0)

    def test_calculate_self_confusion_vector(self):
        confusion_vector = rate_fingerprint.calculate_self_confusion_vector(fingerprint_function=fingerprint_core.fp, weights=np.ones(constants.fingerprint_length),
                                                           distance_function=NNSearch.distance_1_k,
                                                           distance_power=0.5)

        unittest.assertTrue(isinstance(confusion_vector,np.ndarray))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()


