__author__ = 'jeremy'

import unittest
from trendi.paperdoll import neurodoll_falcon_client as nfc



class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')

    def setUp(self):
        self.url='https://s-media-cache-ak0.pinimg.com/236x/df/a3/0a/dfa30af65a46ad8267d148dcefd813d1.jpg'
        pass

    def test_nd_output(self):
        dict = nfc.pd(self.url)
        assert(dict['success'] is not None)
        print('dict from falcon dict:'+str(dict))
        return dict

    def test_nd_categorical_output(self):
        category_index = 0
        dic = nfc.pd(self.url, category_index=category_index)
        assert(dict['success'] is not None)
        if not dic['success']:
            logging.debug('nfc pd not a success')
            return False, []
        return dict

    def test_multilabel_output(self):
        self.url='https://s-media-cache-ak0.pinimg.com/236x/df/a3/0a/dfa30af65a46ad8267d148dcefd813d1.jpg'
        multilabel_dict = nfc.pd(url, get_multilabel_results=True)
        assert(multilabel_dict['success'] is not None)
        print('dict from falcon dict:'+str(multilabel_dict))
        if not multilabel_dict['success']:
            print('did not get nfc pd result succesfully')
            return
        multilabel_output = multilabel_dict['multilabel_output']
        print('multilabel output:'+str(multilabel_output))
        assert(multilabel_output is not None)
        return multilabel_dict #


if __name__ == '__main__':
    unittest.main()


