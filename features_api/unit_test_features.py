import unittest

from trendi.features_api import classifier_client

#for face detection
from trendi.yonatan import cropping


class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')


    def setUp(self):
        self.urls = ['http://i1.adis.ws/s/boohooamplience/azz51307_ms.jpg',
  'https://static.missguided.co.uk/media/catalog/product/cache/3/image/600x870/9df78eab33525d08d6e5fb8d27136e95/h/i/high_neck_lace_dress_kirstie_07.07.16_hm_144201_a.jpg' ]


    def test_features(self):

        features = ['collar','sleeve_length','length','style','dress_texture']
        for url in self.urls:
            for feature in features:
                result = classifier_client.get(feature,url)
                print('result for {} on {} is {}'.format(feature,url,result))
                print


    def test_gender(self):

        urls = ['http://i1.adis.ws/s/boohooamplience/azz51307_ms.jpg',
  'https://static.missguided.co.uk/media/catalog/product/cache/3/image/600x870/9df78eab33525d08d6e5fb8d27136e95/h/i/high_neck_lace_dress_kirstie_07.07.16_hm_144201_a.jpg' ]
        features = ['collar','sleeve_length','length','style','dress_texture']
        for url in self.urls:
            face = cropping.find_that_face(url,1)
            result = classifier_client.get('gender',url,{'face':face})
            print('result for gender on {} is {}'.format(url,result))


if __name__ == "__main__":
    unittest.main()