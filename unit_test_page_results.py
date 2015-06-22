__author__ = 'jeremy'
import unittest

# theirs

#ors
import page_results


class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')

    #def setUp(self):


    def test_verify_hash_of_image(self):
        img_hash = 'f53429c450b2aaecbd2d875b687e09b7'
        image_url = 'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg'
        assert (page_results.verify_hash_of_image(img_hash, image_url))


if __name__ == '__main__':
    unittest.main()


