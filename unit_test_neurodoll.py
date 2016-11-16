__author__ = 'jeremy'

import unittest
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from trendi.paperdoll import neurodoll_falcon_client as nfc
from trendi.classifier_stuff.caffe_nns import jrinfer
from trendi import constants

url='https://s-media-cache-ak0.pinimg.com/236x/df/a3/0a/dfa30af65a46ad8267d148dcefd813d1.jpg'

def test_nd_against_testset(image_and_masks_file='/home/jeremy/image_dbs/colorful_fashion_parsing_data/images_and_labelsfile_test.txt',labels=constants.ultimate_21):
    n_cl = len(labels)
    print('n channels: '+str(n_cl))
    hist = np.zeros((n_cl, n_cl))
    with open(image_and_masks_file,'r') as fp:
        lines =  fp.read().splitlines()
        imagefiles = [s.split()[0] for s in lines]
        labelfiles = [s.split()[1] for s in lines]
        n_files = len(imagefiles)
        for i in range(n_files):
            imfile = imagefiles[i]
            lbfile = labelfiles[i]
            logging.debug('imagefile {} labelfile {}'.format(imfile,lbfile))
            output_dict = nfc.pd(imfile,get_combined_results=True)
            inferred_mask = output_dict['mask']
            gt_mask = cv2.imread(lbfile)
            if len(gt_mask.shape)!=2:
                logging.debug('got weird size mask ({}), using first channel'.format(gt_mask.shape))
                gt_mask = gt_mask[:,:,0]
            confmat = jrinfer.fast_hist(gt_mask.flatten, inferred_mask.flatten(), n_cl)
            hist += confmat
            logging.debug(confmat)
    results_dict = jrinfer.results_from_hist(hist)
    logging.debug(results_dict)
    results_to_html('test.html',results_dict)

def results_to_html(outfilename,results_dict):
    acc = results_dict['class_accuracy']
    overall_acc = results_dict['overall_acc']
    mean_acc = results_dict['mean_acc']
    class_iou = results_dict['class_iou']
    mean_iou = results_dict['mean_iou']
    fwavacc = results_dict['fwavacc']
    with open(outfilename,'a') as f:
        f.write('<br>\n')
        f.write('acc per class:'+ str(acc)+'\n')
        f.write('<br>\n')
        f.write('overall acc:'+ str(overall_acc)+'\n')
        f.write('<br>\n')
        f.write('mean acc:'+ str(mean_acc)+'\n')
        f.write('<br>\n')
        f.write('IU per class:'+ str(class_iou)+'\n')
        f.write('<br>\n')
        f.write('mean IU:'+ str(mean_iou)+'\n')
        f.write('<br>\n')
        f.write('fwavacc:'+ str(fwavacc)+'\n')
        f.write('<br>\n')
        f.write('<br>\n')

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')

    def setUp(self):
        self.url='https://s-media-cache-ak0.pinimg.com/236x/df/a3/0a/dfa30af65a46ad8267d148dcefd813d1.jpg'
        pass

    def test_nd_output(self):
        dict = nfc.pd(self.url)
        assert(dict['success'] is not None)
        print('dict from falcon dict:'+str(dict))
        return dict

    def test_nd_categorical_output(self):
        category_index = 0
        dict = nfc.pd(self.url, category_index=category_index)
        assert(dict['success'] is not None)
        if not dict['success']:
            logging.debug('nfc pd not a success')
            return False, []
        return dict

    def test_multilabel_output(self):
        multilabel_dict = nfc.pd(self.url, get_multilabel_results=True)
        assert(multilabel_dict['success'] is not None)
        print('dict from falcon dict:'+str(multilabel_dict))
        if not multilabel_dict['success']:
            print('did not get nfc pd result succesfully')
            return
        multilabel_output = multilabel_dict['multilabel_output']
        print('multilabel output:'+str(multilabel_output))
        assert(multilabel_output is not None)
        return multilabel_dict #

    def test_combined_output(self):
        output_dict = nfc.pd(self.url,get_combined_results=True)
        assert(output_dict['success'] is not None)
        print('dict from falcon:'+str(output_dict))
        if not output_dict['success']:
            print('did not get nfc pd result succesfully')
            return
        return output_dict #





if __name__ == '__main__':
    unittest.main()


