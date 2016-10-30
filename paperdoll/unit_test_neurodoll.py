__author__ = 'jeremy'



import unittest
import time
import neurodoll_falcon_client as nfc
import numpy as np

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    #    def testPass(self):
    #        return
    #    def testFail(self):
    #        self.failIf(True)
    #    def testError(self):
    #        raise RuntimeError('Test error!')
    #run a timing test



    def test_nd_multilabel_combined(self):


        url = 'http://i.imgur.com/ahFOgkm.jpg'
        print('testing nd on:'+url)
        start_time=time.time()
        results_dict = nfc.pd(url)
        assert(results_dict is not None)
        assert('mask' in results_dict)
        mask = results_dict['mask']
        print('mask shape:'+mask.shape)
        print('unique mask values:'+np.unique(mask))
        print('elapsed:'+str(time.time()-start_time))

        print('testing nd w get_multilabel_results:'+url)
        start_time=time.time()
        multilabel_dict = nfc.pd(url, get_multilabel_results=True)
        assert(multilabel_dict is not None)
        print('multilabel dict:'+multilabel_dict)
        print('elapsed:'+str(time.time()-start_time))


        print('testing nd w multilabel combined results:'+url)
        start_time=time.time()
        multilabel_dict = nfc.pd(url, get_combined_results=True)
        assert(multilabel_dict is not None)
        print('combined dict:'+multilabel_dict)
        print('elapsed:'+str(time.time()-start_time))


        print('testing nd w get all graylevels:'+url)
        start_time=time.time()
        multilabel_dict = nfc.pd(url, get_all_graylevels=True)
        assert(multilabel_dict is not None)
        print('get_all_graylevels dict:'+multilabel_dict)
        print('elapsed:'+str(time.time()-start_time))


    #run a timing test


if __name__ == '__main__':
    unittest.main()
