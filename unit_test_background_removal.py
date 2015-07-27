__author__ = 'jeremy'
__author__ = 'jeremy'
import unittest
import requests
import shutil
import time
import os

import cv2

import background_removal


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

    def test_is_relevant(self):
        url = 'http://i.huffpost.com/gen/1321037/thumbs/o-KATE-LANPHEAR-570.jpg'
        url = 'http://schoolofba.com/wp-content/uploads/2014/09/faces-300x300.jpg'
        url = 'http://img07.deviantart.net/ee95/i/2011/058/8/4/many_faces_of_david_tennant_2_by_pfeifhuhn-d3ag5g6.jpg'
        url = 'http://face-negotiationtheory.weebly.com/uploads/4/2/1/6/4216257/1771161.jpg'
        url = 'http://www.travelmarketing.fr/wp-content/uploads/2014/04/multiple-faces-e1401642770899.jpg'
        url = 'http://www.joancanto.com/blog/wp-content/uploads/2008/07/unai-jvcanto.jpg'
        url = 'http://p1cdn02.thewrap.com/images/2015/01/e-fashion-police-joan-rivers-premiere.jpg'
        file = os.path.join('images', url.split('/')[-1])

        print('file:' + file)
        img = cv2.imread(file)
        if img is None:
            print('img doesnt exists locally, getting from url')
            response = requests.get(url, stream=True)
            with open(file, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
            time.sleep(0.5)
            img = cv2.imread(file)
        if img is None:
            print('cant get image')
            return
        relevance = background_removal.image_is_relevant(img)
        print relevance
        if len(relevance.faces) > 0:
            for face in relevance.faces:
                print('face:' + str(face))
                cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), [0, 255, 0])
            cv2.imshow('faces', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()


