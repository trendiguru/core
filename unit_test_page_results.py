__author__ = 'jeremy'
import unittest

# ours
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

    def test_new_images(self):
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'
        print('testing new images')
        ans = page_results.new_images(page_url, [img_url])
        print('new images results:')
        print ans

    def test_results_for_page(self):
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'

        results_for_page = page_results.results_for_page(page_url)
        print('results for page:')
        print results_for_page


if __name__ == '__main__':
    unittest.main()


