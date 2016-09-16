__author__ = 'jeremy'
import numpy as np
import os
import cv2
import random
import logging
logging.basicConfig(level=logging.DEBUG)
from PIL import Image


from trendi import constants
from trendi.utils import imutils
from trendi import Utils

def write_cats_from_db_to_textfile(image_dir='/home/jeremy/image_dbs/tamara_berg/images',catsfile = 'tb_cats_from_webtool.txt'):
    '''
    for tamara berg cats
    :param image_dir:
    :param catsfile:
    :return:
    '''
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs in db')
    lines_written = 0
    with open(catsfile,'w') as fp:
        for i in range(n_done):
            document = cursor.next()
            url = document['url']
            filename = os.path.basename(url)
            full_path = os.path.join(image_dir,filename)
            items_list = document['items'] #
            hotlist = np.zeros(len(constants.web_tool_categories_v2))
            if not 'already_seen_image_level' in document:
                print('no votes for this doc')
                continue
            if document['already_seen_image_level'] < 2:
                print('not enough votes for this doc')
                continue
            for item in items_list:
                cat = item['category']
                if cat in constants.web_tool_categories_v2:
                    index = constants.web_tool_categories_v2.index(cat)
                elif cat in constants.tamara_berg_to_web_tool_dict:
                    print('WARNING translating from TB')
                    raw_input('WARNING')
                    cat = constants.tamara_berg_to_web_tool_dict[cat]
                    index = constants.web_tool_categories_v2.index(cat)
                else:
                    print('could not figure out this category : '+str(cat))
                    if cat == 'blazer':
                        index = constants.web_tool_categories_v2.index('jacket')
                        print('replacing blazer with jacket ( cat {}) '.format(index))
                    continue
                hotlist[index] = 1
#                print('item:'+str(cat))
            print('hotlist:'+str(hotlist))
            line = str(full_path) +' '+ ' '.join(str(int(n)) for n in hotlist)
            lines_written +=1
            fp.write(line+'\n')
    print(str(lines_written)+' lines written to '+catsfile)

def consistency_check_multilabel_db():
    '''
    read multilabel db, tally up total tags
    check images that have been gone over by 2 or more ppl
    do something about disagreements
    '''
    n_consistent = 0
    n_inconsistent = 0
    db = constants.db
    cursor = db.training_images.find()
    n_total = cursor.count()
    print(str(n_total)+' docs total')
    for document in cursor:
#    for i in range(n_total):
#        document = cursor.next()
#        print(document)
        items_list = document['items']
        if items_list is None:
            print('no items in doc')
            continue
        totlist = {}
        for item in items_list:
            cat = item['category']
#            print('cat:'+str(cat))
            if cat in constants.web_tool_categories_v2 :
#                print('cat in webtool cats v2')
                pass
            elif cat in constants.tamara_berg_to_web_tool_dict:
#                print('cat in tamara_ber_to_webtool_dict')
                pass
            else:
                print('unrecognized cat')
            if cat in totlist:
                totlist[cat] += 1
            else:
                totlist[cat] = 1
        print('totlist:'+str(totlist))
        if totlist == {}:
            print('totlist is {}')
            continue
        cat_totals = [totlist[cat] for cat in totlist]
#        print('cat totals:'+str(cat_totals))
        consistent = cat_totals and all(cat_totals[0] == elem for elem in cat_totals)
        n_consistent = n_consistent + consistent
        n_inconsistent = n_inconsistent + int(not(consistent))
        print('consistent:'+str(consistent)+' n_con:'+str(n_consistent)+' incon:'+str(n_inconsistent))

def binary_pos_and_neg_from_multilabel_db(image_dir='/home/jeremy/image_dbs/tamara_berg_street_to_shop/photos',catsfile_dir = './'):
    '''
    read multilabel db.
    if n_votes[cat] = 0 put that image in negatives for cat.
    if n_votes[cat] = n_voters put that image in positives for cat
    '''
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    print(str(n_done)+' docs done')
    for i in range(n_done):
        document = cursor.next()
        if not 'already_seen_image_level' in document:
            print('no votes for this doc')
            continue
        if document['already_seen_image_level']<2:
            print('not enough votes for this doc')
            continue
        url = document['url']
        filename = os.path.basename(url)
        full_path = os.path.join(image_dir,filename)
        if not os.path.exists(full_path):
            print('file '+full_path+' does not exist, skipping')
            continue
        items_list = document['items'] #
        if items_list is None:
            print('no items in doc')
            continue
        print('items:'+str(items_list))
        votelist = [0]*len(constants.web_tool_categories_v2)
        for item in items_list:
            cat = item['category']
            if cat in constants.web_tool_categories_v2:
                index = constants.web_tool_categories_v2.index(cat)
            elif cat in constants.tamara_berg_to_web_tool_dict:
                print('old cat being translated')
                cat = constants.tamara_berg_to_web_tool_dict[cat]
                index = constants.web_tool_categories.index(cat)
            else:
                print('unrecognized cat')
                continue
            votelist[index] += 1
            print('item:'+str(cat) +' votes:'+str(votelist[index]))
        print('votes:'+str(votelist))
        for i in range(len(votelist)):
            catsfile = os.path.join(catsfile_dir,constants.web_tool_categories_v2[i]+'_filipino_labels.txt')
            print('catsfile:'+catsfile)
            with open(catsfile,'a') as fp:
                if votelist[i]==0:
                    line = str(full_path) + ' 0 \n'
                    print line
                    fp.write(line)
                if votelist[i] >= 2:
                    line = str(full_path) + ' 1 \n'
                    print line
                    fp.write(line)
                fp.close()

def dir_of_dirs_to_labelfiles(dir_of_dirs,class_number=1):
    dirs = [os.path.join(dir_of_dirs,d) for d in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,d))]
    for d in dirs:
        print('doing directory:'+str(d))
        dir_to_labelfile(d,class_number,outfile=os.path.basename(d)+'_labels.txt',filter='.jpg')


def dir_to_labelfile(dir,class_number,outfile='labels.txt',filter='.jpg'):
    '''
    take a dir and add the files therein to a text file with lines like:
    /path/to/file class_number
    :param dir:
    :param class_number: assign all files this class #
    :param outfile : write to this file.  Appends, doesn't overwrite
    :return:
    '''
    if filter:
        files=[os.path.join(dir,f) for f in os.listdir(dir) if filter in f]
    else:
        files=[os.path.join(dir,f) for f in os.listdir(dir)]
    i = 0
    with open(outfile,'a') as fp:
        for f in files:
            line = f + ' '+str(class_number)
            print line
            fp.write(line+'\n')
            i+=1
        fp.close()
    print(str(i)+' images written to '+outfile+' with label '+str(class_number))

def copy_negatives(filename = 'tb_cats_from_webtool.txt',outfile =  None):
    '''
    file lines are of the form /path/to/file class_number
    :param filename:
    :return:
    '''
    negs = []
    if outfile == None:
        outfile = filename[:-4]+'_negs.txt'
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            if cat == 0:
                negs.append(line)
        fp.close()
    print('n_negatives {}'.format(len(negs)))

    with open(outfile,'w') as fp:
        for line in negs:
            fp.write(line)

def inspect_category_textfile(filename = 'tb_cats_from_webtool.txt',n_cats=None,visual_output=False):
    '''
    file lines are of the form /path/to/file class_number
    :param filename:
    :return:
    '''
    if not n_cats:
        n_cats = len(constants.web_tool_categories_v2)
    n_instances = [0]*n_cats
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            n_instances[cat]+=1
        fp.close()

    print('n_instances {}'.format(n_instances))

    if visual_output:
        with open(filename,'r') as fp:
            for line in fp:
                print line
                path = line.split()[0]
                cat = int(line.split()[1])
                print(cat)
    #            im = Image.open(path)
    #            im.show()
                img_arr = cv2.imread(path)
                imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)

def inspect_multilabel_textfile(filename = 'tb_cats_from_webtool.txt'):
    '''
    for 'multi-hot' labels of the form 0 0 1 0 0 1 0 1
    so file lines are /path/to/file 0 0 1 0 0 1 0 1
    :param filename:
    :return:
    '''
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

def inspect_pixlevel_textfile(filename = 'images_and_labelsfile.txt'):
    with open(filename,'r') as fp:
        for line in fp:
            print line
            path1 = line.split()[0]
            img_arr = cv2.imread(path1)
            cv2.imshow('image',img_arr)

            path2 = line.split()[1]
            imutils.show_mask_with_labels(path2,labels=constants.ultimate_21,visual_output=True)

def split_to_trainfile_and_testfile(filename='tb_cats_from_webtool.txt', fraction=0.05):
    with open(filename,'r') as fp:
        lines = fp.readlines()
        print('file {} has lines like {}'.format(filename,lines[0]))
        random.shuffle(lines)
        n_lines = len(lines)
        train_lines = lines[0:int(n_lines*(1-fraction))]
        test_lines = lines[int(n_lines*(1-fraction)):]
        train_name = filename[0:-4] + '_train.txt'
        test_name = filename[0:-4] + '_test.txt'
        print('{} files written to {} and {} files written to {}'.format(len(train_lines),train_name,len(test_lines),test_name))
        with open(train_name,'w') as trfp:
            trfp.writelines(train_lines)
            trfp.close()
        with open(test_name,'w') as tefp:
            tefp.writelines(test_lines)
            tefp.close()

def balance_cats(filename='tb_cats_from_webtool.txt', fraction=0.5,n_cats=2,outfilename='tb_cats_balanced.txt'):
    '''
    balance the occurence of categories - take minimum occurences and let all cats occur only that amt
    ie. if there are 10 examples of class 1, 20 examples class 2, 30 examples class 3, take examples of each class and write
    to outfilename
    there is a theorectical question here of whether this is desireable or not
    :param filename: input file with lines of the form '/path/to/file  class_number'
    :param fraction:
    :return:
    '''
    n_instances = [0]*n_cats
    instances = None*n_cats #iniitialize in Nones . there seems to be no oneliner like instances = [] * n_cats
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            n_instances[cat]+=1
            instances[cat].append(line)
            print('path {} cat {} n_instances {}'.format(path,cat,n_instances,instances))
        fp.close()
    min_instances = min(n_instances)

    #kill the initial Nones
    for i in range(n_cats):
        del(instances[i][0])
#  a shuffle cant hurt here
    with open(outfilename,'w') as fp:
        for i in range(n_cats):
            for j in range(min_instances):
                fp.write(instances[cat][j])
    fp.close()


def textfile_for_pixlevel(imagesdir,labelsdir=None,imagefilter='.jpg',labelsuffix='.png', outfilename = None):
    if labelsdir == None:
        labelsdir = imagesdir
    if outfilename == None:
        outfilename = os.path.join(imagesdir,'images_and_labelsfile.txt')
    imagefiles = [f for f in os.listdir(imagesdir) if imagefilter in f]
    print(str(len(imagefiles))+' imagefiles found in '+imagesdir)
    with open(outfilename,'w') as fp:
        for f in imagefiles:
            labelfile = f[:-4]+labelsuffix
            labelfile = os.path.join(labelsdir,labelfile)
            if not os.path.exists(labelfile):
                logging.debug('could not find labelfile {} corresponding to imagefile {}'.format(labelfile,f))
                continue
            imagefile = os.path.join(imagesdir,f)
            line = imagefile +' '+ labelfile
            print('writing: '+line)
            fp.write(line+'\n')

def textfile_for_pixlevel_kaggle(imagesdir,labelsdir=None,imagefilter='.tif',labelsuffix='_mask.tif', outfilename = None):
    if labelsdir == None:
        labelsdir = imagesdir
        imagefiles = [f for f in os.listdir(imagesdir) if imagefilter in f and not labelsuffix in f]
    else:
        imagefiles = [f for f in os.listdir(imagesdir) if imagefilter in f]

    if outfilename == None:
        outfilename = os.path.join(imagesdir,'images_and_labelsfile.txt')
    print(str(len(imagefiles))+' imagefiles found in '+imagesdir)
    with open(outfilename,'w') as fp:
        for f in imagefiles:
            labelfile = f[:-4]+labelsuffix
            labelfile = os.path.join(labelsdir,labelfile)
            if not os.path.exists(labelfile):
                logging.debug('could not find labelfile {} corresponding to imagefile {}'.format(labelfile,f))
                continue
            imagefile = os.path.join(imagesdir,f)
            line = imagefile +' '+ labelfile
            print('writing: '+line)
            fp.write(line+'\n')


if __name__ == "__main__": #
#    write_cats_from_db_to_textfile()
#    split_to_trainfile_and_testfile()
#    inspect_textfile()

#test_u21_256x256_no_aug

    dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/'
    textfile_for_pixlevel(imagesdir=dir+'images/train_u21_256x256_no_aug',labelsdir=dir+'labels_256x256',outfilename=dir+'images_and_labelsfile_train.txt')
#    split_to_trainfile_and_testfile(dir+'images_and_labelsfile.txt')
#    inspect_pixlevel_textfile(dir+'images_and_labelsfile_train.txt')

    textfile_for_pixlevel(imagesdir=dir+'images/test_u21_256x256_no_aug',labelsdir=dir+'labels_256x256',outfilename=dir+'images_and_labelsfile_test.txt')
#    split_to_trainfile_and_testfile(dir+'images_and_labelsfile.txt')
    inspect_pixlevel_textfile(dir+'images_and_labelsfile_test.txt')