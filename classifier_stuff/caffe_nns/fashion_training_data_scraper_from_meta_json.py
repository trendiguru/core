__author__ = 'natanel'

import os
import json
import urllib
import urllib2
from joblib import Parallel, delayed
import multiprocessing
import cv2
import time
import logging
import traceback
import sys
import shutil

from trendi import Utils
from trendi.classifier_stuff import darknet_convert
from trendi import constants
import numpy as np

logging.basicConfig(level=logging.DEBUG)


# getting the links and image numbers to web links to list from the text:
#example line for product #23  , url starts at char 11
# 000000023,http://media1.modcloth.com/community_outfit_image/000/000/097/900/img_full_4c2895555fe8.jpg

def rename_bbox_files(dir):
    only_files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)) and 'bbox' in f ]
    for a_file in only_files:
#example:        product_5651_photo_2177_bbox_664_571_1160_2291.jpg
        full_source = os.path.join(dir,a_file)
        dest_split = a_file.split('_bbox_')[0]
        elements = dest_split.split('_')
        product = elements[1]
        photo = elements[3]
        dest_base = 'photo_'+photo+'.jpg'
        print('photo {} prod {} dest {} name {}'.format(photo, product, dest_base, a_file))
#        shutil.move(full_source, dest)
        dest = os.path.join(dir,dest_base)
        shutil.copy(full_source, dest)

def get_product_photos(images_files_path):
    images_links = open(images_files_path)
    listing = []
    for line in images_links:
        if line[-1:] == '\n':
            listing.append(line[10:-1])
        else:
            listing.append(line[10:])
    print len(listing)
    return listing

def dl_photos(photosfile,images_savedir,start_from=0):
    listings = get_product_photos(photosfile)
    max_items = 5000000
    Utils.ensure_dir(images_savedir)
    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores)(delayed(get_file)(listings[i],i+1,images_savedir) for i in range(start_from,len(listings)))

#    for i in range(len(listings)):

 #       get_file(listings[i],i+1,images_savedir)


def get_file(url,n_photo,write_dir):
    fname = 'photo_'+str(n_photo)+'.jpg'
    fname = os.path.join(write_dir,fname)
    try:
        url_call = urllib2.urlopen(url,data='',timeout = 1000)
        f = open(fname, 'wb')
        f.write(url_call.read())
        f.flush()
        f.close()
        print('saved '+fname)
    except:
        print('failed to get '+fname)
# for json_file in only_files:
# instead of a for loop, lets parallelize! :
def library_for_dataset_scraping(json_file, photos_path,max_items):
    # finds only dresses dataset:
    data = []

    # folder creation for each json file packet:
    set_name = json_file[:-5]
    print('jsonfile: {0} photopath: {1}'.format(json_file,photos_path))
    if not os.path.exists(photos_path ):
        os.mkdir(photos_path)

    # making sure only json files are read:
    if json_file[-4:] == 'json':
        data = json.load(open(json_file))
        n = 0
        for data_pack in data:
            photo_id = data_pack['photo']
            product_id = data_pack['product']
            # annotated data ordering (if exists)
            if len(data_pack) > 2:
                bbox_dict = data_pack['bbox']
                bbox = [int(bbox_dict['left']), int(bbox_dict['top']), int(bbox_dict['width']), int(bbox_dict['height'])]
#                file_name = 'product_%s_photo_%s_bbox_%s_%s_%s_%s.jpg' % (product_id, photo_id, bbox[0], bbox[1], bbox[2], bbox[3])
#                file_name = 'product_%s_photo_%s.jpg' % (product_id, photo_id)
                file_name = 'photo_%s.jpg' % (photo_id)
            else:
                file_name = 'photo_%s.jpg' % (photo_id) #
#               file_name = 'product_%s_photo_%s.jpg' % (product_id, photo_id)
            # downloading the images from the web:
            fname = os.path.join(photos_path,file_name)
            f = open(fname, 'wb')
            try:
                url_call = urllib.urlopen(listing[photo_id-1])
                f.write(url_call.read())
                f.close()
                print fname + ' saved'
            except:
                print fname + ' passed'
                pass
            n+=1
            if n>max_items:
                print('hit max items')
                break
    else:
        print('not a json file')


#TODO write file to check for same product number in different dirs and combine the bb file
#DONE
def combine_bbs(json_dir,imagefiles_dir_of_dirs):
    json_files = [f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir,f)) and f[-5:] == '.json' and not 'retrieval' in f]
    print('json files:'+str(json_files))
    n = 0
    for json_file in json_files:
        print('file:'+str(json_file))
        prefix = json_file[:-5]
        imagefiles_dir = prefix
        fullpath = os.path.join(json_dir,json_file)
        data = json.load(open(fullpath))
        l = len(data)
        print('length = '+str(l))
        combine_bbs_within_category(data)

#effort abandoned in light of better wa yto do it, with single bbfile per image
def combine_bbs_within_category(data):	
    for i in range(0,len(data)):
#    for data_pack in data:
        data_pack = data[i]
        photo_id = data_pack['photo']
        product_id = data_pack['product']
        print('photo:{} product:{}'.format(photo_id,product_id))
        if not 'bbox' in data_pack:
            logging.warning('bbox data not found')
            continue
        bbox_dict = data_pack['bbox']
        bbox = [int(bbox_dict['left']), int(bbox_dict['top']), int(bbox_dict['width']), int(bbox_dict['height'])]
        file_name = 'product_%s_photo_%s_bbox_%s_%s_%s_%s.jpg' % (product_id, photo_id, bbox[0], bbox[1], bbox[2], bbox[3])
    for j in range(i,len(data)):
        data_pack2 = data[i]
        photo_id2 = data_pack['photo']
        product_id2 = data_pack['product']
      #      print('photo2:{} product2:{}'.format(photo_id2,product_id2))
        if not 'bbox' in data_pack2:
            logging.warning('bbox data not found')
            continue
        if photo_id2 == photo_id:
            pass


def generate_bbfiles_from_dir_of_jsons(json_dir,imagefiles_dir_of_dirs,bb_dir,darknet=True,category_number=None,clobber=False):
#    only_dirs = [d for d in json_dir if os.path.isdir(os.path.join(json_dir,d))]
    json_files = [f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir,f)) and f[-5:] == '.json' and not 'retrieval' in f]
    class_number = 0
    Utils.ensure_dir(bb_dir)
    for a_file in json_files:
        prefix = a_file[0:-5]
        imagefiles_dir = prefix
        print('json {} images {} class {}'.format(a_file,imagefiles_dir,class_number))
        full_json_file = os.path.join(json_dir,a_file)
        full_images_dir = os.path.join(imagefiles_dir_of_dirs,prefix)
        generate_bbfiles_from_json(full_json_file,full_images_dir,bb_dir,darknet=True,class_number=class_number,clobber=False)
        class_number += 1

def generate_bbfiles_from_json(json_file,imagefiles_dir,bb_dir,darknet=True,class_number=None,clobber=False):
    '''
    This is to take a json file from tamara berg stuff and write a file having the bb coords,
    either in darknet (percent) or pixel format.
    Currently all bbs of a given image are entered into the same bbfile
    :param json_file:
    :param listing:
    :param max_items:
    :param docrop:
    :return:
    '''
    data = []
    # making sure onlu json files are read:
    if json_file[-4:] != 'json':
        logging.warning('nonjson file sent')
        return
    print('jsonfile:{} imagesdir:{}'.format(json_file,imagefiles_dir))
  #  raw_input('enter to continue')
    parent_dir = Utils.parent_dir(json_file)
    data = json.load(open(json_file))
    n = 0
    for data_pack in data:
        photo_id = data_pack['photo']
        product_id = data_pack['product']
        print('photo:{} product:{}'.format(photo_id,product_id))
        if not 'bbox' in data_pack:
            logging.warning('bbox data not found')
            continue
        bbox_dict = data_pack['bbox']
        bbox = [int(bbox_dict['left']), int(bbox_dict['top']), int(bbox_dict['width']), int(bbox_dict['height'])]
#        bbfilebase = file_name[0:-4]   #file.jpg -> file
        bbfilebase = str(photo_id)   #file.jpg -> file
        bbfile = bbfilebase+'.txt'
#        cropped_name = 'product_%s_photo_%s_cropped.jpg' % (product_id, photo_id)
#        full_path = photos_path + set_name + '/' + file_name
#        cropped_path = photos_path + set_name + '/' + cropped_name
        bb_full_filename = os.path.join(bb_dir,bbfile)
#        print  'attempting full+cropped img save of: ' + full_path
        if not clobber:
            f = open(bb_full_filename, 'a')  #append to end of bbfile to allow for multiple bbs for same file 'a+' is read/appendwrite
        else:
            f = open(bb_full_filename, 'w')  #overwrite bbfile

        # consider that same image may occur in several dirs....
    #considered!
        try:
            if darknet:
                file_name = 'product_%s_photo_%s_bbox_%s_%s_%s_%s.jpg' % (product_id, photo_id, bbox[0], bbox[1], bbox[2], bbox[3])
                full_filename = os.path.join(imagefiles_dir,file_name)
                print('looking for file '+full_filename)
                if os.path.isfile(full_filename):
                    img_arr = cv2.imread(full_filename)
                    if img_arr is None:
                        logging.warning('could not open file '+str(full_filename))
                        continue
                    h,w = img_arr.shape[0:2]
                    dark_bbox = darknet_convert.convert((w,h),bbox)
                    print('bb {} darkbb{} w {} h {}'.format(bbox,dark_bbox,w,h))
                    bbox = dark_bbox
                else:
                    print('could not find file')
                    continue
                n_digits = 5
                line_to_write = str(class_number)+' '+str(bbox[0])[0:n_digits]+' '+str(bbox[1])[0:n_digits]+' '+str(bbox[2])[0:n_digits]+' '+str(bbox[3])[0:n_digits] + '\n'
                f.write(line_to_write)
                f.flush()
                print 'successful dark_bb appended to  ' + bb_full_filename
     #                   print listing[photo_id-1] + '\n cropped succesful save as: ' + cropped_path
            else: #write  in regular format (class, x y w  h  in pixels)
                line_to_write = str(class_number)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3]) + '\n'
                f.write(line_to_write)
                f.flush()
                print 'successful pixel bb appended to  ' + bb_full_filename

        except Exception as e:
            print(str(traceback.format_exception(*sys.exc_info())))
            raise
    f.close()

def library_with_cropping(json_file, json_files_path, photos_path, listing,max_items, docrop=False):
    # finds only dresses dataset:
    data = []

    # folder creation for each json file packet:
    set_name = json_file[:-5]
    print('jsonfile: {0} photopath: {1}'.format(json_file,photos_path))
    if not os.path.exists(photos_path + set_name):
        os.mkdir(photos_path + set_name)

    # making sure onlu json files are read:
    if json_file[-4:] == 'json':
        data = json.load(open(json_files_path + json_file))
        n = 0
        for data_pack in data:
            print('')
            photo_id = data_pack['photo']
            product_id = data_pack['product']
            # annotated data ordering (if exists)
            # product# , photo#, bb
            if len(data_pack) > 2:
                file_name = 'product_%s_photo_%s_withbb.jpg' % (product_id, photo_id)
                bbox_dict = data_pack['bbox']
                bbox = [int(bbox_dict['left']), int(bbox_dict['top']), int(bbox_dict['width']), int(bbox_dict['height'])]
                file_name = 'product_%s_photo_%s_bbox_%s_%s_%s_%s.jpg' % (product_id, photo_id, bbox[0], bbox[1], bbox[2], bbox[3])
                cropped_name = 'product_%s_photo_%s_cropped.jpg' % (product_id, photo_id)
                full_path = photos_path + set_name + '/' + file_name
                cropped_path = photos_path + set_name + '/' + cropped_name
                print  'attempting full+cropped img save of: ' + full_path
                f = open(full_path, 'wb')
                try:
                    url_call = urllib.urlopen(listing[photo_id-1])
                    f.write(url_call.read())
                    f.flush()
                    f.close()
                    print listing[photo_id-1] + '\n succesful full img saved as: ' + full_path
      #              time.sleep(0.1)
                    if docrop:
                        try:
                            print('trying crop')
                            img_arr = cv2.imread(full_path)
                            if img_arr is None:
                                print('could not read img:'+full_path)
                            cropped = img_arr[bbox[1]:bbox[3]+bbox[1],bbox[0]:bbox[2]+bbox[0]]
                            cropped_path = full_path  #clobber orig image
                            cv2.imwrite(cropped_path,cropped)
                            print listing[photo_id-1] + '\n cropped succesful save as: ' + cropped_path
                        except:
                            print listing[photo_id-1] + '\n cropped unsuccesful save as: ' + cropped_path
                except:
                    print listing[photo_id-1] + '\n yo full img unsuccesful save of: ' + full_path

            # product# , photo#, no bb
            else:
                file_name = 'product_%s_photo_%s.jpg' % (product_id, photo_id)
                full_path = photos_path + set_name + '/' + file_name
                print  'attempting full catalog img save of: ' + full_path
                f = open(full_path, 'wb')
                try:
                    url_call = urllib.urlopen(listing[photo_id-1])
                    f.write(url_call.read())
                    f.close()
                    print listing[photo_id-1] + '\n catalog img saved as: ' + full_path
                except:
                    print listing[photo_id-1] + '\n catalog img unsuccesful save: ' + full_path + '\n'
                    pass
            n+=1
            print('n='+str(n))

            if n>max_items:
                print('hit max items')
                break
    else:
        print('not a json file')

def generate_bbfiles_from_json_dir_of_dirs(dir_of_jsons,imagefiles_dir,bb_dir,darknet=False,positive_filter=None):
    initial_only_files = [dir for dir in os.listdir(dir_of_jsons) if os.path.isfile(os.path.join(dir_of_jsons,dir)) ]
    initial_only_files.sort()
    only_files = []
    if positive_filter:
        for a_file in initial_only_files:
            if positive_filter in a_file:
                only_files.append(a_file)
    else:
        only_files = initial_only_files

    category_number = 0
    print('outer dir of jsons:{} '.format(dir_of_jsons))
    print('json files {}'.format(only_files))
    for a_file in only_files:
        fullfile = os.path.join(dir_of_jsons,a_file)
        #only take 'test' or 'train' dirs, if test_or_train is specified
        print('doing file {} class {} ({})'.format(fullfile,category_number,constants.tamara_berg_categories[category_number]))
        raw_input('ret to cont')
        generate_bbfiles_from_json(fullfile,imagefiles_dir,bb_dir,darknet=False,class_number = category_number,clobber=False)
        category_number += 1
#def generate_bbfiles_from_json(json_file,imagefiles_dir,bb_dir,darknet=True,class_number=None,clobber=False):

def tamara_berg_to_ultimate_21(tb_index):
    u21s = [conv[1] for conv in constants.tamara_berg_to_ultimate_21_index_conversion if conv[0]==tb_index]
    if len(u21s) != 1:
        logging.warning('could not get u21 index for tamaraberg index '+str(tb_index))
        return None
    return int(u21s[0])

def multi_class_labels_from_bbfiles(dir_of_bbfiles):
#json files are in this format:
#  {"photo": 417, "product": 2400, "bbox": {"width": 145, "top": 405, "left": 390, "height": 235}}
    with open('classlabels.txt','w') as classfile:
        files = [f for f in os.listdir(dir_of_bbfiles) if '.txt' in f]
        print(str(len(files))+' files in '+dir_of_bbfiles)
        for a_file in files:
            with open(os.path.join(dir_of_bbfiles,a_file),'r') as lines:
                outvec = np.zeros(len(constants.ultimate_21))
                for line in lines:
      #              print('file:'+a_file+' line:'+str(line))
                    line_arr = line.split()
                    berg_class = int(line_arr[0])
                    u21_class = tamara_berg_to_ultimate_21(berg_class)
                    if u21_class is None:
                        logging.warning('no matching class found')
                        continue
                    print('berg_class {} ({}) u21_class {} ({}) line {}'.format(berg_class,constants.tamara_berg_categories[berg_class],u21_class,constants.ultimate_21[u21_class],line_arr))
                    outvec[u21_class] = 1
            imgname = '/home/jeremy/image_dbs/tamara_berg/images/photo_'+a_file[:-4]+'.jpg'
            print('imgname:'+imgname)
            if os.path.exists(imgname):
                img_arr = cv2.imread(imgname)
                h,w=img_arr.shape[0:2]
                factor = float(h)/400.0
                resized = cv2.resize(img_arr,(int(w/factor),400))
                cv2.imshow('win',resized)
                cv2.waitKey(0)
            else:
                print('image not found')
            writeline = a_file+' '
            for i in range(len(outvec)):
                writeline = writeline+str(int(outvec[i]))+' '
            print('writing line:'+str(writeline))
            classfile.write(writeline)
#            raw_input('enter to continue)')



if __name__ == "__main__":
# opening the JSONs structure files:
    num_cores = multiprocessing.cpu_count()
  #  num_cores = 1
    json_files_path = os.path.dirname(os.path.abspath(__file__)) + '/meta/json/'
    json_dir = '/home/jeremy/street2shop/meta/json/'
    json_file = '/home/jeremy/dataset/json/test_pairs_bags.json'
    json_file = '/home/jeremy/street2shop/meta/json/test_pairs_bags.json'
    bb_dir = '/home/jeremy/'
    imagefiles_dir = '/home/jeremy/dataset/test_pairs_bags'
    imagefiles_dir = '/home/jeremy/street2shop/photos/'

#    dir_of_bbfiles = '/home/jeremy/image_dbs/tamara_berg/'
#    multi_class_labels_from_bbfiles(dir_of_bbfiles)

    json_files_path = '/home/jeremy/image_dbs/tamara_berg/dataset/json'
    photos_path = '/home/jeremy/image_dbs/tamara_berg/new_photos'
    jsons = [f for f in os.listdir(json_files_path) if 'json' in f]

    dl_photos('/home/jeremy/image_dbs/tamara_berg/dataset/photos/photos.txt',photos_path)

 #   for json_file in jsons:
 #       library_for_dataset_scraping(os.path.join(json_files_path,json_file), photos_path,max_items=1000000)

    if(0):
        generate_bbfiles_from_json_dir_of_dirs(json_dir,imagefiles_dir,bb_dir,darknet=True,positive_filter='train')
        generate_bbfiles_from_json_dir_of_dirs(json_dir,imagefiles_dir,bb_dir,darknet=True,positive_filter='test')
    #    generate_bbfiles_from_json(json_file,imagefiles_dir,darknet=True,class_number=66)



    if(0):
        images_files_path = os.path.dirname(os.path.abspath(__file__)) + '/photos/photos.txt'
        dl_path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
        if not os.path.exists(dl_path):
                os.mkdir(dl_path)

        listing = get_product_photos(images_files_path)
        max_items = 5000000
        only_files = [f for f in os.listdir(json_files_path) if os.path.isfile(os.path.join(json_files_path, f))]
    #    only_files = ['test_pairs_footwear.json']
        print only_files
        Parallel(n_jobs=num_cores)(delayed(library_with_cropping)(json_file,json_files_path,dl_path,listing,max_items) for json_file in only_files)
