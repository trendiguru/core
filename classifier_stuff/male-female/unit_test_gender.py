__author__ = 'jeremy'

# TODO in     def test_get_all_data_for_page(self): - add multiple images for one page url and make sure both get returned

__author__ = 'jeremy'
import unittest
import trendi_guru_modules.gender

# ours
#import page_results
import dbUtils


class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')

    def test_gender(self):
        img_name = 'test/uncentered/female1.jpg'
        g = gender(img_name)
        print(g)
        img_name = 'test/uncentered/male1.jpg'
        g = gender(img_name)
        print(g)
        img_name = 'test/uncentered/male2.jpg'
        g = gender(img_name)
        print(g)
    #WIP

if __name__ == '__main__':
    unittest.main()



