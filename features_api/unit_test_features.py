import unittest

from trendi.features_api import classifier_client
from trendi import Utils

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
                'http://www.betterthanpants.com/media/catalog/product/e/x/extreme-hunting-cool-shirt-female-model.jpg'

    def test_features(self):
        features = ['collar','sleeve_length','length','style','dress_texture']
        for url in self.urls:
            for feature in features:
                result = classifier_client.get(feature,url)
                print('result for {} on {} is {}'.format(feature,url,result))
                print


    def test_gender(self):
      for url in self.urls:
            img_arr = Utils.get_cv2_img_array(url)
            if img_arr is None:
                continue
            print('image size:'+str(img_arr.shape))
            face_dict = cropping.find_that_face(img_arr,1)
            face = face_dict['faces'][0]
            print('face x,y,w,h: '+str(face))
            result = classifier_client.get('gender',url,url=url,face=face)
            print('result for gender on {} is {}'.format(url,result))


if __name__ == "__main__":
    unittest.main()