__author__ = 'jeremy'
__author__ = 'jeremy'
import unittest
import cv2

from trendi.utils import imutils

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

    def test_do_resize(self):
        path = trendi.__file__
        img_arr = cv2.imread(os.path.join(path,'images/female1.jpg')
        desired_size = (200,300)
        resized = imutils.resize_keep_aspect(img_arr,output_size=desired_size)
        actual_size=resized.shape[0:2]
        assert(actual_size==desired_size)
        print('orig size {} new size {}'.format(img_arr.shape,resized.shape))

    def test_undo_resize(self):
        img_arr = cv2.imread('../images/female1.jpg')
        desired_size = (250,351)
        resized = imutils.undo_resize_keep_aspect(img_arr, output_file=None, output_size = desired_size,use_visual_output=True,careful_with_the_labels=True)
        print('orig size {} new size {}'.format(img_arr.shape,resized.shape))
        actual_size=resized.shape[0:2]
        assert(actual_size==desired_size)

#        resize_keep_aspect(infile, output_file=output_file, output_size = (600,401),use_visual_output=True)
#        undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)

#        resize_keep_aspect(infile, output_file=output_file, output_size = (600,399),use_visual_output=True)
#        undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)

#        resize_keep_aspect(infile, output_file=output_file, output_size = (400,600),use_visual_output=True)
#        undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)

##        resize_keep_aspect(infile, output_file=output_file, output_size = (400,601),use_visual_output=True)
 #       undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)

 #       resize_keep_aspect(infile, output_file=output_file, output_size = (400,599),use_visual_output=True)
 #       undo_resize_keep_aspect(output_file, output_file=None, output_size = (orig_h,orig_w),use_visual_output=True,careful_with_the_labels=True)

if __name__ == '__main__':
    unittest.main()


