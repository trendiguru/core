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
import sys

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

#binary lists generated so far (9.10.16)
#dress
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

def create_class_a_vs_class_b_file_from_multilabel_db(index_a,index_b,image_dir='/home/jeremy/image_dbs/tamara_berg_street_to_shop/photos',outfile=None,labels=constants.web_tool_categories_v2):
    '''
    read multilabel db.
    if n_votes[cat] = 0 put that image in negatives for cat.
    if n_votes[cat] = n_voters put that image in positives for cat
    '''
    if outfile is None:
        outfile = 'class'+str(index_a)+'_vs_class'+str(index_b)+'.txt'
    db = constants.db
    cursor = db.training_images.find()
    n_done = cursor.count()
    n_instances=[0,0]
    output_cat_for_a = 0
    output_cat_for_b = 1
    outlines=[]
    print(str(n_done)+' docs to check')
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
#        print('items:'+str(items_list))
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
#            print('item:'+str(cat) +' votes:'+str(votelist[index]))
        print('votes:'+str(votelist))
        if votelist[index_a]>=2 and votelist[index_b]==0:
            line = str(full_path) + ' '+str(output_cat_for_a)+'\n'
            n_instances[0]+=1
            print('catA file {} n {}'.format(full_path,n_instances))
            outlines.append(line)
        elif votelist[index_a]==0 and votelist[index_b]>=2:
            line = str(full_path) + ' '+str(output_cat_for_b)+'\n'
            n_instances[1]+=1
            print('catB file {} n {}'.format(full_path,n_instances))
            outlines.append(line)
        else:
            print('{} votes for cat {} and {} votes for cat {} b, not using'.format(votelist[index_a],index_a,votelist[index_b],index_b))
    print('writing {} lines to {}, breakdown:{}'.format(len(outlines),outfile,n_instances))
    with open(outfile,'w') as fp:
        for l in outlines:
            fp.write(l)
        fp.close()

def create_class_a_vs_class_b_file_from_multilabel_file(index_a,index_b,multilabel_textfile,visual_output=False,outfile=None):
    if outfile is None:
        outfile = 'class'+str(index_a)+'_vs_class'+str(index_b)+'.txt'
    n_instances=[0,0]
    output_cat_for_a = 0
    output_cat_for_b = 1
    outlines = []
    with open(multilabel_textfile,'r') as fp:
        for line in fp:
   #         print line
            path = line.split()[0]
            vals = [int(v) for v in line.split()[1:]]
            v1 = vals[index_a]
            v2 = vals[index_b]
            if v1 and v2:
                print('got image {} with both cats, not using'.format(path))
            elif v1:
                n_instances[0]+=1
                outlines.append(path+' '+str(output_cat_for_a))
                print('indexa {} indexb {} file {} n {}'.format(v1,v2,path,n_instances))
            elif v2:
                n_instances[1]+=1
                outlines.append(path+' '+str(output_cat_for_b))
                print('indexa {} indexb {} file {} n {}'.format(v1,v2,path,n_instances))
            else:
                print('got image {} with no cats, not using'.format(path))
            if(visual_output):
                img_arr = cv2.imread(path)
                imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)
        fp.close()
    with open(outfile,'a') as f2:
        for line in outlines:
            f2.write(line+'\n')
        f2.close()


def dir_of_dirs_to_labelfiles(dir_of_dirs,class_number=1):
    dirs = [os.path.join(dir_of_dirs,d) for d in os.listdir(dir_of_dirs) if os.path.isdir(os.path.join(dir_of_dirs,d))]
    for d in dirs:
        print('doing directory:'+str(d))
        dir_to_labelfile(d,class_number,outfile=os.path.basename(d)+'_labels.txt',filter='.jpg')

def dir_to_labelfile(dir,class_number,outfile=None,filter='.jpg'):
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
    if outfile == None:
        outfile = os.path.join(dir,'labelfile.txt')
    with open(outfile,'a') as fp:
        for f in files:
            line = f + ' '+str(class_number)
            print line
            fp.write(line+'\n')
            i+=1
        fp.close()
    print('added {} files to {} with class {}'.format(len(files),outfile,class_number))
    print('used {} files from dir with {} files'.format(len(files),len(os.listdir(dir))))
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

def inspect_single_label_textfile(filename = 'tb_cats_from_webtool.txt',n_cats=None,visual_output=False,randomize=False):
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
    if randomize:
        random.shuffle(lines)
    n = 0
    n_encountered = [0]*n_cats
    if visual_output:
        for line in lines:
            n = n + 1
            print line
            path = line.split()[0]
            cat = int(line.split()[1])
            n_encountered[cat]+=1
            print(str(n)+' images seen, totals:'+str(n_encountered))
#            im = Image.open(path)
#            im.show()
            img_arr = cv2.imread(path)
            imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)

def inspect_multilabel_textfile(filename = 'tb_cats_from_webtool.txt',labels=constants.web_tool_categories_v2):
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
                    cats = cats + ',' + labels[i]
            print(cats)
            print()
            img_arr = cv2.imread(path)
            if img_arr is None:
                print('could not grok file '+path)
                continue
            imutils.resize_to_max_sidelength(img_arr, max_sidelength=250,use_visual_output=True)

def inspect_pixlevel_textfile(filename = 'images_and_labelsfile.txt',labels=constants.ultimate_21):
    with open(filename,'r') as fp:
        for line in fp:
            print line
            path1 = line.split()[0]
#            img_arr = cv2.imread(path1)
#            cv2.imshow('image',img_arr)

            path2 = line.split()[1]
            imutils.show_mask_with_labels(path2,labels=labels,original_image=path1,visual_output=True)

def split_to_trainfile_and_testfile(filename='tb_cats_from_webtool.txt', fraction=0.05):
    '''
    writes (destructively) files with _train.txt and _test.txt based on filename, with sizes determined by fraction
    :param filename: input catsfile
    :param fraction: ratio test:train
    :return:
    '''
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

def balance_cats(filename='tb_cats_from_webtool.txt', ratio_neg_pos=2.0,n_cats=2,outfilename=None,shuffle=True):
    '''
    balance the occurence of categories - take minimum occurences and let all cats occur only that amt
    ie. if there are 10 examples of class 1, 20 examples class 2, 30 examples class 3, take 10 examples of each class and write
    to outfilename
    there is a theorectical question here of whether this is desireable or not (maybe unbalanced is good if wild is unbalanced)
    this works only for 2 cats (todo - make it work for n cats).  also , assumes there are more negs than pos
    :param filename: input file with lines of the form '/path/to/file  class_number'
    :param ratio_neg_pos: number of negs vs. positives to include , n_neg = n_pos*ratio_neg_pos
    :param outfilename file to write to, if not given writes to original path of catsfile.txt but with filename catsfile.balanced.txt
    :param n_cats not implemented , assumes n_cats=2
    :param shuffle not implemented
    :return:
    '''
    print('balancing '+filename+' with ratio '+str(ratio_neg_pos)+', '+str(n_cats)+' categories')
    n_instances = [0]*n_cats
    instances = []  #*n_cats#iniitialize in Nones . there seems to be no oneliner like instances = [] * n_cats
    for i in range(n_cats):
        instances.append([])
    with open(filename,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            path = line.split()[0]
            cat = int(line.split()[1])
            try:
                n_instances[cat]+=1
            except:
                print "Unexpected error:", sys.exc_info()[0]
                print('trying to parse line:')
                print(line)
                print('cat = '+str(cat))
                continue
            instances[cat].append(line)
#        print('path {} cat {} n_instances {}'.format(path,cat,n_instances,instances))
        fp.close()
        print('n_instances {}'.format(n_instances))
    n_negs = n_instances[0]
    n_pos = n_instances[1]
    min_instances = min(n_instances)
    desired_negs = (n_pos*ratio_neg_pos)
    negs_to_use = int(min(desired_negs,n_negs))
    #kill the initial Nones
#    for i in range(n_cats):
#        del(instances[i][0])
#  a shuffle cant hurt here
    if outfilename is None:
        outfilename = filename.replace('.txt','')+'_balanced.txt'
    print('writing {} positives and {} negatives to {}'.format(n_pos,negs_to_use,outfilename))
#    if(shuffle):
#        instances
    with open(outfilename,'w') as fp:
        for i in range(n_cats):
            if i==1:
                jrange=min_instances
            else:
                jrange=negs_to_use
            for j in range(jrange):
                fp.write(instances[i][j])
            print('wrote '+str(jrange)+' lines for category '+str(i))
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


#
if __name__ == "__main__": #
#    write_cats_from_db_to_textfile()
#    split_to_trainfile_and_testfile()
#    inspect_textfile()

#test_u21_256x256_no_aug
#    dir_to_file_singlelabel(dir,classindex,labelfile,outfile=None,filter='.jpg'):
#    balance_cats(f)
#    outfilename = f.replace('.txt','')+'_balanced.txt'
#    split_to_trainfile_and_testfile(outfilename)


    '''x = ['bag_filipino_labels.txt',
         'belt_filipino_labels.txt',
         'bracelet_filipino_labels.txt',
         'cardigan_filipino_labels.txt',
         'coat_filipino_labels.txt',
         'dress_filipino_labels.txt',
         'earrings_filipino_labels.txt',
         'eyewear_filipino_labels.txt',
         'footwear_filipino_labels.txt',
         'hat_filipino_labels.txt',
         'jacket_filipino_labels.txt',
         'jeans_filipino_labels.txt',
         'necklace_filipino_labels.txt',
         'overalls_filipino_labels.txt',
         'pants_filipino_labels.txt',
         'scarf_filipino_labels.txt',
         'shorts_filipino_labels.txt',
         'skirt_filipino_labels.txt',
         'stocking_filipino_labels.txt',
         'suit_filipino_labels.txt',
         'sweater_filipino_labels.txt',
         'sweatshirt_filipino_labels.txt',
         'top_filipino_labels.txt',
         'watch_filipino_labels.txt',
         'womens_swimwear_bikini_filipino_labels.txt',
         'womens_swimwear_nonbikini_filipino_labels.txt']
    dir = '/home/jeremy/image_dbs/tamara_berg_street_to_shop/todo/'
    x = [os.path.join(dir,f) for f in os.listdir(dir) if '.txt' in f]
    x.sort()
    for f in x:
        balance_cats(f)
        outfilename = f.replace('.txt','')+'_balanced.txt'
        split_to_trainfile_and_testfile(outfilename)
'''
## change from photos to photos_250x250:
#sed s'/photos/photos_250x250/' bag_filipino_labels_balanced.txt > bag_filipino_labels_250x250.txt

    if(0):
        dir = '/home/jeremy/image_dbs/colorful_fashion_parsing_data/'
        textfile_for_pixlevel(imagesdir=dir+'images/train_u21_256x256_no_aug',labelsdir=dir+'labels_256x256',outfilename=dir+'images_and_labelsfile_train.txt')
    #    split_to_trainfile_and_testfile(dir+'images_and_labelsfile.txt')
    #    inspect_pixlevel_textfile(dir+'images_and_labelsfile_train.txt')

        textfile_for_pixlevel(imagesdir=dir+'images/test_u21_256x256_no_aug',labelsdir=dir+'labels_256x256',outfilename=dir+'images_and_labelsfile_test.txt')
    #    split_to_trainfile_and_testfile(dir+'images_and_labelsfile.txt')
        inspect_pixlevel_textfile(dir+'images_and_labelsfile_test.txt')


        #useful script - change all photos to photos_250x250
#!/usr/bin/env bash
#echo $1
#name=$(echo $1|sed 's/.txt/_250x250.txt/')
#echo $name
#sed 's/photos/photos_250x250/' $1 > $name

#use with
#    for f in *.txt; do ./doit.sh $f; done
