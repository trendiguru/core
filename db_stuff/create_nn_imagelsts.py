__author__ = 'jeremy'
import numpy as np
import os
import cv2
import random
import logging
logging.basicConfig(level=logging.DEBUG)

from trendi import constants
from trendi.utils import imutils

def write_cats_from_db_to_textfile(image_dir='/home/jeremy/image_dbs/tamara_berg/images',catsfile = 'tb_cats_from_webtool.txt'):
    '''
    for tamara berg cats
    :param image_dir:
    :param catsfile:
    :return:
    '''
    db = constants.db
    cursor = db.training_images.find({'already_done':True})
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    with open(catsfile,'w') as fp:
        for i in range(n_done):
            document = cursor.next()
            url = document['url']
            filename = os.path.basename(url)
            full_path = os.path.join(image_dir,filename)
            items_list = document['items'] #
            hotlist = np.zeros(len(constants.web_tool_categories))
            for item in items_list:
                cat = item['category']
                if cat in constants.web_tool_categories:
                    index = constants.web_tool_categories.index(cat)
                else:
                    if cat in constants.tamara_berg_to_web_tool_dict:
                        cat = constants.tamara_berg_to_web_tool_dict[cat]
                        index = constants.web_tool_categories.index(cat)
                hotlist[index] = 1
                print('item:'+str(cat))
            print('hotlist:'+str(hotlist))
            line = str(full_path) +' '+ ' '.join(str(int(n)) for n in hotlist)
            fp.write(line+'\n')

def inspect_textfile(filename = 'tb_cats_from_webtool.txt'):
    with open(filename,'r') as fp:
        for line in fp:
            print line
            path = line.split()[0]
            cats = ''
            for i in range(len(constants.web_tool_categories)):
                current_val = int(line.split()[i+1])
#                print('cur digit {} val {}'.format(i,current_val))
                if current_val:
                    cats = cats + ',' + constants.web_tool_categories[i]
            print(cats)
            print()
            img_arr = cv2.imread(path)
            imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)


def split_to_trainfile_and_testfile(filename='tb_cats_from_webtool.txt', fraction=0.05):
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            print line
        print lines[0]
        random.shuffle(lines)
        print lines[0]
        n_lines = len(lines)
        train_lines = lines[0:int(n_lines*(1-fraction))]
        test_lines = lines[int(n_lines*(1-fraction)):]
        print('{} trainingfiles and {} testingfiles'.format(len(train_lines),len(test_lines)))
        train_name = filename[0:-4] + '_train.txt'
        test_name = filename[0:-4] + '_test.txt'
        with open(train_name,'w') as trfp:
            trfp.writelines(train_lines)

        with open(test_name,'w') as trfp:
            trfp.writelines(test_lines)


def textfile_for_pixlevel(imagesdir,labelsdir=None,imagefilter='.jpg',labelsuffix='.png', outfilename = None):
    if labelsdir == None:
        labelsdir = imagesdir
    if outfilename == None:
        outfilename = os.path.join(imagesdir,'images_and_labelsfile.txt')
    imagefiles = [f for f in os.listdir(dir) if imagefilter in f]
    with open(outfilename,'w'):
    for f in imagefiles:
        labelfile = f[:-4]+labelsuffix
        labelfile = os.path.join(labelsdir,labelfile)
        if not os.exists(labelfile):
            logging.debug('could not find labelfile {} corresponding to imagefile {}'.format(labelfile,f))

    os.path.join(dir,f)

if __name__ == "__main__": #
    write_cats_from_db_to_textfile()
    split_to_trainfile_and_testfile()
    inspect_textfile()