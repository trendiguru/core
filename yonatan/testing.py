#!/usr/bin/env python

__author__ = 'yonatan_guy'

import re


# yonatan_txt_file = open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth_yonatan.txt', 'w')
#
# with open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth.txt', 'r') as handle:
#     for line in handle:
#         # print line.split('  ')
#         # print re.sub(r'([^\s])\s([^\s])', r'\1_\2', line)
#         yonatan_txt_file.write(re.sub(r'([^\s])\s([^\s])', r'\1_\2', line))



yonatan_txt_file_dict = open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth_yonatan_dict.txt', 'w')

with open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth.txt',
          'r') as handle:
    for count, line in enumerate(handle):
        words = line.split()
        # print re.sub(r'([^\s])\s([^\s])', r'\1_\2', line)
        # yonatan_txt_file.write(re.sub(r'([^\s])\s([^\s])', r'\1_\2', line))

        yonatan_txt_file_dict.write("\"" + count + "\": [\'" + words[0] + "\', \'" + words[1] + "\']")
