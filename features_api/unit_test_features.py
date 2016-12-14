import unittest

from trendi.features_api import classifier_client

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')


    def setUp(self):

        pass


    def test_features(self):

        urls = ['http://i1.adis.ws/s/boohooamplience/azz51307_ms.jpg',
  'https://static.missguided.co.uk/media/catalog/product/cache/3/image/600x870/9df78eab33525d08d6e5fb8d27136e95/h/i/high_neck_lace_dress_kirstie_07.07.16_hm_144201_a.jpg' ]
        features = ['collar','sleeve_length','length','style','dress_texture','gender']
        for url in urls:
            for feature in features:
                result = classifier_client.get(feature,url)
                print('result for {} on {} is {}'.format(feature,url,result))

if __name__ == "__main__":
    unittest.main()