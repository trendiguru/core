#!/usr/bin/env python

__author__ = 'yonatan_guy'





with open('/data/jeremy/image_dbs/deep_fashion/category_and_attribute_prediction/list_attr_cloth.txt', 'r') as handle:
    for line in handle:
        print line.split('  ')


# train_text_file = open("/home/yonatan/faces_stuff/55k_face_train_list.txt", "r")
#
#
# train_text_file.write("\"" + i + "\": [\'" + attribute + "\', \'" + type + "\']")
