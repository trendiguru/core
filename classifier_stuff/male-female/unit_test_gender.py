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

    def test_gender_func(self):
        img_name = 'images/female1.jpg'
        g = genderfunc(img_name)
        print('trying genderfunc:'+ img_name+' seems to be '+ str(g))

        img_name = 'images/male1.jpg'
        g = genderfunc(img_name)
        print('trying genderfunc:' + img_name+' seems to be '+ str(g))

        img_name = 'images/male2.jpg'
        g = genderfunc(img_name)
        print('trying genderfunc:' + img_name+' seems to be '+ str(g))

        img_name = 'images/male3.jpg'
        g = genderfunc(img_name)
        print('trying genderfunc:' + img_name+' seems to be '+ str(g))

    def test_gender_queue(self):
        img_name = 'images/female1.jpg'
        g = gender(img_name)
        print(img_name+' seems to be '+ str(g))

        img_name = 'images/male1.jpg'
        g = gender(img_name)
        print('trying gender queue:' + img_name+' seems to be '+ str(g))

        img_name = 'images/male2.jpg'
        g = gender(img_name)
        print('trying gender queue:' + img_name+' seems to be '+ str(g))

        img_name = 'images/male3.jpg'
        g = gender(img_name)
        print('trying gender queue:' + img_name+' seems to be '+ str(g))


    #WIP

if __name__ == '__main__':
    unittest.main()



