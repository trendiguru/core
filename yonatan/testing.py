#!/usr/bin/env python

__author__ = 'yonatan_guy'

import re






with open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth.txt', 'r') as handle:
    for line in handle:
        # print line.split('  ')
        match = re.match(r"([0-9])", line, re.I)
        if match:
            print 'YES'
            items = match.groups()
            print items
        else:
            print "NO"

# attribute_txt_file = open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth.txt', 'r')






# train_text_file.write("\"" + i + "\": [\'" + attribute + "\', \'" + type + "\']")
