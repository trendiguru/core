__author__ = 'jeremy'

# TODO
#add tests for:
#make sure fingerprint of simlar images are more similar than fps of dvery different images
#length is correct

import unittest
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

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


    def test_fp(self):

        # answer should be a dictionary of info about bb or an error string if no bb found
        # url = 'http://lp.hm.com/hmprod?set=key[source],value[/model/2014/3PV%200235738%20001%2087%206181.jpg]&set=key[rotate],value[]&set=key[width],value[]&set=key[height],value[]&set=key[x],value[]&set=key[y],value[]&set=key[type],value[STILL_LIFE_FRONT]&hmver=4&call=url[file:/product/large]'
        urls = ['http://img.sheinside.com/images/sheinside.com/201403/1395131162147422866.jpg']
        urls.append('http://cdn.iwastesomuchtime.com/1072012024419MsOiX.jpg')
        urls.append('https://s-media-cache-ak0.pinimg.com/236x/04/08/b6/0408b6b4f14fa1ac31f3e649beeffbb0.jpg')
        urls.append('http://www.polyvore.com/cgi/img-thing?.out=jpg&size=l&tid=17842198')

        filename = 'images/img.jpg'
        img_arr = Utils.get_cv2_img_array(filename)
        sh = img_arr.shape
        print('shape is ' + str(sh))
        if img_arr is not None:
                mask = np.ones((img_arr.shape[0], img_arr.shape[1]), np.uint8)
                fingerprint = fingerprint_core.fp(img_arr)
                self.assertTrue(len(fingerprint) > 0)

        else:
            print('couldnt get image:'+str(url))

            # raw_input('hit return for next')

if __name__ == '__main__':
    unittest.main()
