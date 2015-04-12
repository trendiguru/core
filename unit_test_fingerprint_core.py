__author__ = 'jeremy'

import unittest
import pymongo

import Utils
import fingerprint_core
import constants
import cv2
import numpy as np

fingerprint_length = constants.fingerprint_length


class OutcomesTest(unittest.TestCase):
    # examples of things to return
    #    def testPass(self):
    #        return

    #    def testFail(self):
    #        self.failIf(True)

    #    def testError(self):
    #        raise RuntimeError('Test error!')


    def test_show_fp(self):
    #answer should be a dictionary of info about bb or an error string if no bb found
        url = 'http://lp.hm.com/hmprod?set=key[source],value[/model/2014/3PV%200235738%20001%2087%206181.jpg]&set=key[rotate],value[]&set=key[width],value[]&set=key[height],value[]&set=key[x],value[]&set=key[y],value[]&set=key[type],value[STILL_LIFE_FRONT]&hmver=4&call=url[file:/product/large]'
        img_arr = Utils.get_cv2_img_array(url)
        if img_arr is not None:
                cv2.imshow('im1',img_arr)
                k=cv2.waitKey(50)& 0xFF
                fingerprint = fingerprint_core.fp(img_arr, bounding_box=None, weights = np.ones(fingerprint_length))
                return_val = fingerprint_core.show_fp(fingerprint)
                self.assertTrue(return_val)

        else:
            print('couldnt get image:'+str(url))



if __name__ == '__main__':
    unittest.main()
