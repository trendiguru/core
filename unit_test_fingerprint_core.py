__author__ = 'jeremy'

import unittest
import math

import cv2
import numpy as np

import Utils
import fingerprint_core

import constants


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

        # answer should be a dictionary of info about bb or an error string if no bb found
        # url = 'http://lp.hm.com/hmprod?set=key[source],value[/model/2014/3PV%200235738%20001%2087%206181.jpg]&set=key[rotate],value[]&set=key[width],value[]&set=key[height],value[]&set=key[x],value[]&set=key[y],value[]&set=key[type],value[STILL_LIFE_FRONT]&hmver=4&call=url[file:/product/large]'

        url = 'http://cdn.iwastesomuchtime.com/1072012024419MsOiX.jpg'
        img_arr = Utils.get_cv2_img_array(url)
        if img_arr is not None:
                cv2.imshow('im1',img_arr)
                k=cv2.waitKey(50)& 0xFF
                mask = np.ones((img_arr.shape()))
                fingerprint = fingerprint_core.fp(img_arr, mask)
                return_val = fingerprint_core.show_fp(fingerprint)
                for i in range(0,1000000):
                    a=math.cos(math.sin(i))
                self.assertTrue(return_val)

        else:
            print('couldnt get image:'+str(url))

if __name__ == '__main__':
    unittest.main()
