# TODO in     def test_get_all_data_for_page(self): - add multiple images for one page url and make sure both get returned

__author__ = 'jeremy'
import unittest



# ours
import page_results
import dbUtils
from .constants import db

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')

    def setUp(self):
        # self.connection = Connection('localhost', 27017)  # Connect to mongodb
        #print(self.connection.database_names())  # Return a list of db, equal to: > show dbs
        #self.db = self.connection['mydb']  # equal to: > use testdb1
        self.db = db
        print(self.db.collection_names())  # Return a list of collections in 'testdb1'
        print("images exists in db.collection_names()?")  # Check if collection "posts"
        print("images" in self.db.collection_names())  # Check if collection "posts"
        # exists in db (testdb1
        collection = self.db['images']
        print('collection.count() == 0 ?' + str(collection.count() == 0))  # Check if collection named 'posts' is empty


    def test_verify_hash_of_image(self):
        print('starting test of verify_hash_of_image()')
        img_hash = 'f53429c450b2aaecbd2d875b687e09b7'
        image_url = 'http://resources.shopstyle.com/pim/c8/af/c8af6068982f408205491817fe4cad5d.jpg'
        assert (page_results.verify_hash_of_image(img_hash, image_url))

    def test_new_images(self):
        print('starting test of new_images()')
        img_url1 = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        img_url1 = 'http://p1cdn02.thewrap.com/images/2015/01/e-fashion-police-joan-rivers-premiere.jpg'
        img_url2 = 'http://ohwtflol.com/wp-content/uploads/2014/08/Coolest-kid-ever-Why-did-my-parents-not-think-of-dressing-me-like-this-when-I-was-young.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'
        page_url = 'testurl1.com'
        page_url = 'http://ohwtflol.com/coolest-kid-ever/'
        page_url = 'http://www.thewrap.com/es-fashion-police-fiasco-what-went-wrong-what-happens-next/'
        print('testing new images')
        n_found, n_not_found = page_results.new_images(page_url, [img_url1, img_url2])
        print('images found in db:' + str(n_found) + ' and images not found:' + str(n_not_found))

    # OK
    def test_get_all_data_for_page(self):  # TODO - add multiple images for one page url and make sure both get returned
        print('starting test of get_all_data_for_page()')

        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'

        img_url = 'http://p1cdn02.thewrap.com/images/2015/01/e-fashion-police-joan-rivers-premiere.jpg'
        page_url = 'http://www.thewrap.com/es-fashion-police-fiasco-what-went-wrong-what-happens-next/'

        results_for_page = page_results.get_all_data_for_page(page_url)
        print('results for page:')
        print results_for_page

        page_url = 'http://ohwtflol.com/coolest-kid-ever/'
        results_for_page = page_results.get_all_data_for_page(page_url)
        print('results for page:')
        print results_for_page

    def test_get_data_for_specific_image(self):
        print('starting test of get_data_for_specific image()')
        img_url = 'http://ohwtflol.com/wp-content/uploads/2014/08/Coolest-kid-ever-Why-did-my-parents-not-think-of-dressing-me-like-this-when-I-was-young.jpg'
        img_url = 'http://p1cdn02.thewrap.com/images/2015/01/e-fashion-police-joan-rivers-premiere.jpg'
        img_hash = 'b2cfe8cae069add119c4a61b05eb39d7'
        results = page_results.get_data_for_specific_image(image_url=img_url, image_hash=None)
        print('results for img url:')
        print str(results)
        self.assertTrue(len(results) > 0)
        results = page_results.get_data_for_specific_image(image_url=None, image_hash=img_hash)
        print('results for img hash:')
        print str(results)
        # self.assertTrue(len(results) > 0)

        img_hash = '6da2c32d55016564ab6f012da77ebcbc'
        results = page_results.get_data_for_specific_image(image_url=None, image_hash=img_hash)
        print('results for img hash:')
        print str(results)
        # self.assertTrue(len(results) > 0)

        img_hash = 'nonexistent_hash'
        results = page_results.get_data_for_specific_image(image_url=None, image_hash=img_hash)
        print('results for img hash:')
        print str(results)
        self.assertTrue(results is None)

        img_url = 'nonexistent_url'
        results = page_results.get_data_for_specific_image(image_url=img_url, image_hash=None)
        print('results for img hash:')
        print str(results)
        self.assertTrue(results is None)

    #OK
    def test_start_pipeline(self):
        print('starting test of start_pipeline()')
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        img_url = 'http://p1cdn02.thewrap.com/images/2015/01/e-fashion-police-joan-rivers-premiere.jpg'
        db_results = page_results.start_pipeline(img_url)
        self.assertTrue(len(db_results) > 1)
        print('results from start_pipeline():')
        print(str(db_results))

    #OK
    def test_find_similar_items_and_put_into_db(self):
        img_url = 'http://resources.shopstyle.com/xim/b7/ce/b7ce6784ec5e488fbe51bc939ce6e1a5.jpg'
        page_url = 'http://www.shopstyle.com/browse/womens-tech-accessories/Salvatore-Ferragamo?pid=uid900-25284470-95'
        img_url = 'http://p1cdn02.thewrap.com/images/2015/01/e-fashion-police-joan-rivers-premiere.jpg'
        page_url = 'http://www.thewrap.com/es-fashion-police-fiasco-what-went-wrong-what-happens-next/'
        page_results.find_similar_items_and_put_into_db(img_url, page_url)

    # OK
    def test_step_thru_images_db(self):
        dbUtils.step_thru_images_db(use_visual_output=True, collection='images')

if __name__ == '__main__':
    unittest.main()


