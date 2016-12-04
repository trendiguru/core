#!/usr/bin/env python

import os

# source_dir = '/home/yonatan/dress_length_from_my_pc'


def edit_dirs_names(source_dir):

    for root, dirs, files in os.walk(source_dir):
        for dir in dirs:
            all_words = dir.split()
            new_dir_name = all_words[0]

            for i in range(1, len(all_words)):
                if all_words[i] == "-":
                    print "{0} -> {1}".format(dir, new_dir_name)
                    break
                new_dir_name = new_dir_name + "_" + all_words[i]
            # os.rename(dir, new_dir_name)
            print "root: {0}, dir: {1}".format(root, dir)

    print "Done editing dirs names"

