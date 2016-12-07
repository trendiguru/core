__author__ = 'jeremy'
import cPickle
import os


def checkout_pkl_file(thefile):
    if os.path.exists(thefile):
        with open(thefile, 'rb') as fid:
            flip_maskdb = cPickle.load(fid)
    for l in flip_maskdb:
        print l