__author__ = 'natanel'

import os
import json
import urllib
from joblib import Parallel, delayed
import multiprocessing
import cv2
import time

# getting the links and image numbers to web links to list from the text:
#example line for product #23  , url starts at char 11
# 000000023,http://media1.modcloth.com/community_outfit_image/000/000/097/900/img_full_4c2895555fe8.jpg

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


# for json_file in only_files:
# instead of a for loop, lets parallelize! :
def library_for_dataset_scraping(json_file,json_files_path, photos_path,max_items):
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
            photo_id = data_pack['photo']
            product_id = data_pack['product']
            # annotated data ordering (if exists)
            if len(data_pack) > 2:
                bbox_dict = data_pack['bbox']
                bbox = [int(bbox_dict['left']), int(bbox_dict['top']), int(bbox_dict['width']), int(bbox_dict['height'])]
                file_name = 'product_%s_photo_%s_bbox_%s_%s_%s_%s.jpg' % (product_id, photo_id, bbox[0], bbox[1], bbox[2], bbox[3])
#                file_name = 'product_%s_photo_%s.jpg' % (product_id, photo_id)
            else:
                file_name = 'product_%s_photo_%s.jpg' % (product_id, photo_id)
            # downloading the images from the web:
            f = open(photos_path + set_name + '/' + file_name, 'wb')
            try:
                url_call = urllib.urlopen(listing[photo_id-1])
                f.write(url_call.read())
                f.close()
                print listing[photo_id-1] + '\n saved as: ' + file_name
            except:
                print listing[photo_id-1] + '\n passed: ' + file_name + '\n'
                pass
            n+=1
            if n>max_items:
                print('hit max items')
                break
    else:
        print('not a json file')

def library_with_cropping(json_file,json_files_path, photos_path,listing,max_items,docrop=False):
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


if __name__ == "__main__":
# opening the JSONs structure files:
    num_cores = multiprocessing.cpu_count()
  #  num_cores = 1
    json_files_path = os.path.dirname(os.path.abspath(__file__)) + '/meta/json/'
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
