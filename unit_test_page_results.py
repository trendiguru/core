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
        print('starting test of verify_hash_of_image()')
        img_hash = 'f53429c450b2aaecbd2d875b687e09b7'
        image_url = 'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg'
        assert (page_results.verify_hash_of_image(img_hash, image_url))

    def test_new_images(self):
        print('starting test of new_images()')
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'
        page_url = 'testurl1.com'
        page_url = 'http://ohwtflol.com/coolest-kid-ever/'
        img_url = 'http://ohwtflol.com/wp-content/uploads/2014/08/Coolest-kid-ever-Why-did-my-parents-not-think-of-dressing-me-like-this-when-I-was-young.jpg'
        print('testing new images')
        n_found, n_not_found = page_results.new_images(page_url, [img_url])
        print('images found in db:' + str(n_found) + ' and ' + str(n_not_found) + ' not found')

    def test_results_for_page(self):
        print('starting test of results_for_page()')
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'

        results_for_page = page_results.results_for_page(page_url)
        print('results for page:')
        print results_for_page

    def test_start_pipeline(self):
        print('starting test of start_pipeline()')
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        db_results = page_results.start_pipeline(img_url)
        self.assertTrue(len(db_results) > 1)
        print('results from start_pipeline():')
        print(str(db_results))

if __name__ == '__main__':
    unittest.main()


