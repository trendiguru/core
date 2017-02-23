__author__ = 'jeremy'

import cv2
import pymongo

from trendi.paperdoll import pd_falcon_client
from trendi import constants
from trendi import Utils
from trendi.utils import imutils


def get_pd_results():
    url = 'https://thechive.files.wordpress.com/2017/02/0c7bf9a4951ade636082e45849b01cd8.jpeg'
    resp = pd_falcon_client.pd(url)
    print('resp:'+str(resp))

def get_groundtruth_for_tamaraberg_multilabel(labelfile='/data/jeremy/image_dbs/labels/labelfiles_tb/tb_cats_from_webtool_round2_train.txt',
                                              label_cats=constants.web_tool_categories_v2):
    with open(labelfile,'r') as fp:
        lines = fp.readlines()
    imgs_and_labels = [(line.split()[0],[int(i) for i in line.split()[1:]]) for line in lines]
    print imgs_and_labels[0]
    print(str(len(imgs_and_labels))+' images described in file '+labelfile)

def dl_images(source_domain='stylebook.de',text_filter='',dl_dir='/data/jeremy/image_dbs/golden/',in_docker=True,visual_output=False):
    '''
    dl everything in the images db, on the assumption that these are the  most relevant to test our answers to
    :return:
    '''

    if in_docker:
        db = pymongo.MongoClient('localhost',port=27017).mydb
    else:
        db = constants.db

    all = db.images.find({'domain':source_domain})
    for doc in all:
        url=doc['image_urls'][0]
        if text_filter in url[0]:
            print url
            Utils.get_cv2_img_array(url,convert_url_to_local_filename=True,download=True,download_directory=dl_dir)
        else:
            print('skipping '+url)

    #move the images with more than one person
    imutils.do_for_all_files_in_dir(imutils.one_person_per_image,dl_dir,visual_output=False)