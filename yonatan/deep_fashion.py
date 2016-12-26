#!/usr/bin/env python

__author__ = 'yonatan_guy'

import numpy as np
from trendi.yonatan import yonatan_constants


def inspect_multilabel_textfile(filename = 'tb_cats_from_webtool.txt'):
    '''
    for 'multi-hot' labels of the form 0 0 1 0 0 1 0 1
    so file lines are /path/to/file 0 0 1 0 0 1 0 1
    :param filename:
    :return:
    '''
    with open(filename,'r') as fp:
        for count, line in enumerate(fp):
            print line
            path = line.split()[0]
            vals = [int(i) for i in line.split()[1:]]
            non_zero_idx = np.nonzero(vals)
            print non_zero_idx
            for i in range(len(non_zero_idx[0])):
                print yonatan_constants.attribute_type_dict[str(non_zero_idx[0][i])]

                # img_arr = cv2.imread(os.path.join("/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction", path))
                # if img_arr is None:
                #     print('could not grok file '+path)
                #     continue

                # cv2.imshow("image", img_arr)
                # cv2.waitKey(0)