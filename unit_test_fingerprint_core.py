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


    def test_show_fp(self):

        # answer should be a dictionary of info about bb or an error string if no bb found
        # url = 'http://lp.hm.com/hmprod?set=key[source],value[/model/2014/3PV%200235738%20001%2087%206181.jpg]&set=key[rotate],value[]&set=key[width],value[]&set=key[height],value[]&set=key[x],value[]&set=key[y],value[]&set=key[type],value[STILL_LIFE_FRONT]&hmver=4&call=url[file:/product/large]'
        url = 'http://img.sheinside.com/images/sheinside.com/201403/1395131162147422866.jpg'
        url = 'http://i.dailymail.co.uk/i/pix/2014/08/16/article-2726891-2094C69B00000578-330_634x947.jpg'
        img_arr = Utils.get_cv2_img_array(url)
        sh = img_arr.shape
        print('shape is ' + str(sh))
        if img_arr is not None:
                cv2.imshow('im1',img_arr)
                k = cv2.waitKey(1000) & 0xFF
                mask = np.ones((img_arr.shape[0], img_arr.shape[1]), np.uint8)
                fingerprint = fingerprint_core.fp(img_arr, mask)

                return_val = fingerprint_core.show_fp(fingerprint)
                for i in range(0,1000000):
                    a=math.cos(math.sin(i))
                self.assertTrue(return_val)

        else:
            print('couldnt get image:'+str(url))


    def test_fp_with_bwg(self):
        urls = ['http://img.sheinside.com/images/sheinside.com/201403/1395131162147422866.jpg']
        urls.append('http://cdn.iwastesomuchtime.com/1072012024419MsOiX.jpg')
        urls.append('https://s-media-cache-ak0.pinimg.com/236x/04/08/b6/0408b6b4f14fa1ac31f3e649beeffbb0.jpg')
        urls.append('http://www.polyvore.com/cgi/img-thing?.out=jpg&size=l&tid=17842198')

        for url in urls:
            img_arr = Utils.get_cv2_img_array(url, download=True)
            fp = fingerprint_core.fp_with_bwg(img_arr)  # with black, white, gray
            fingerprint_core.show_fp(fp)
            cv2.imshow('im1', img_arr)
            k = cv2.waitKey(0)

    def test_eq_RGB(self):
        urls = ['http://img.sheinside.com/images/sheinside.com/201403/1395131162147422866.jpg']
        urls.append('http://cdn.iwastesomuchtime.com/1072012024419MsOiX.jpg')
        urls.append('https://s-media-cache-ak0.pinimg.com/236x/04/08/b6/0408b6b4f14fa1ac31f3e649beeffbb0.jpg')
        urls.append('http://www.polyvore.com/cgi/img-thing?.out=jpg&size=l&tid=17842198')

        for url in urls:
            img_arr = Utils.get_cv2_img_array(url, convert_url_to_local_filename=True, download=True)
            cv2.imshow('im1', img_arr)
            k = cv2.waitKey(50)
            equalized = fingerprint_core.eq_BGR(img_arr)  # with black, white, gray
            cv2.imshow('eq', equalized)
            k = cv2.waitKey(50)
            hist = cv2.calcHist([img_arr], [2], None, [256], [0, 256])
            hist2 = cv2.calcHist([equalized], [2], None, [256], [0, 256])
            plt.plot(hist, 'r')
            plt.plot(hist2, 'g')
            # there's probably something more intelligent to check here....
            self.assertTrue(len(hist2) == len(hist))
            plt.show(block=True)
            # raw_input('hit return for next')

if __name__ == '__main__':
    unittest.main()
