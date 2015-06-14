from __future__ import with_statement

__author__ = 'dr. groovemaster'


import numpy as np
import cv2

GREEN = [0, 255, 0]
RED = [0, 0, 255]
BLUE = [255, 0, 0]

__author__ = 'dr. juice-man'

import os

import pymongo
import Utils
import background_removal
from find_similar_mongo import get_all_subcategories


def dl_keyword_images(category_id, total=2000, keyword=None,
                      # dir='/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/images',
                      # dir='/home/ubuntu/Dev/trendi_guru_modules/images',
                      dir='./images',
                      show_visual_output=False):
    db = pymongo.MongoClient().mydb
    query = {"categories": {"$elemMatch": {"id": {"$in": get_all_subcategories(db.categories, category_id)}}}}
    if keyword is None:
        path = os.path.join(dir, category_id)
        cursor = db.products.find(query)
    else:
        path = os.path.join(dir, category_id)
        path = os.path.join(path, keyword)
        cursor = db.products.find({'$and': [{"description": {'$regex': keyword}}, query]})
    print('path:' + path)
    if not os.path.exists(path):
        print('creating dir')
        os.makedirs(path)
    i = 0
    for item in cursor:
        if i > total:
            break
        i += 1

        item_image = Utils.get_cv2_img_array(item['image']['sizes']['XLarge']['url'])
        if item_image is None:
            return None
        if show_visual_output == True:
            cv2.imshow('im1', item_image)
            k = cv2.waitKey(200)

        if background_removal.image_is_relevant(background_removal.standard_resize(item_image, 400)[0]):
            name = os.path.join(path, str(item['id']) + '.jpg')
            try:
                print('writing ' + name)
                cv2.imwrite(name, item_image)
            except:
                print('couldnt write file:' + name)

def get_items(categories, keywords):
    max_items_per_category = 2000
    for cat in cats:
        if keywords is not None:
            for keyword in keywords:
                print('getting cat ' + cat + ' w keyword ' + keyword)
                dl_keyword_images(cat, total=max_items_per_category, keyword=keyword, show_visual_output=False)
        else:
            print('getting cat ' + cat + ' no keyword ')
            dl_keyword_images(cat, total=max_items_per_category, show_visual_output=False)


def write_bbfile(fp, bb, filename):
    string = filename + ' 1 {0} {1} {2} {3} \n'.format(bb[0], bb[1], bb[2], bb[3])
    print('writing ' + str(string))
    fp.write(string)


def read_and_show_bbfile(bbfilename, parent_dir):
    try:
        with open(bbfilename, 'r') as fp:
            for line in fp:
                values = line.split()
                fname = values[0]
                fname = os.path.join(parent_dir, fname)
                bb = []
                for value in values[2:]:
                    bb.append(int(value))
                print('fname:' + fname + ' bb:' + str(bb))
                img_array = cv2.imread(fname)
                if img_array is None:
                    print('no image gotten, None returned')
                    continue
                else:
                    print('succesfully got ' + fname)
                    cv2.rectangle(img_array, (bb[0], bb[1]),
                                  (bb[0] + bb[2], bb[1] + bb[3]),
                                  GREEN, thickness=1)
                    print('bb=' + str(bb))
                    cv2.imshow('win', img_array)
                    k = cv2.waitKey(200)
                    cv2.destroyAllWindows()
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        print 'oops'


def read_bbs_in_subdirs(parent_dir='images'):
    for dir, subdir_list, file_list in os.walk(parent_dir):
        print('Found directory: %s' % dir)
        bbfilename = os.path.join(dir, 'bbs.txt')
        read_and_show_bbfile(bbfilename, dir)


def box_images(parent_dir='images', use_visual_output=False):
    for dir, subdir_list, file_list in os.walk(parent_dir):
        print('Found directory: %s' % dir)
        bbfilename = os.path.join(dir, 'bbs.txt')
        with open(bbfilename, 'w+') as fp:
            for fname in file_list:
                print('\t%s' % fname)
                full_filename = os.path.join(dir, fname)
                # fp.write

                img_array = cv2.imread(full_filename)
                if img_array is None:
                    print('no image gotten, None returned')
                    continue
                    # elif not isinstance(img_array[0][0], int):
                    # print('no image gotten, not int')
                    #             continue
                else:
                    print('succesfully got ' + full_filename)
                    bb = get_bb(img_array, use_visual_output, fname=fname)
                    print('bb=' + str(bb) + ' x1y1x2y2:' + str(bb[0]) + ',' + str(bb[1]) + ',' + str(
                        bb[0] + bb[2]) + ',' + str(bb[1] + bb[3]))
                    if bb is not None:
                        write_bbfile(fp, bb, fname)
                        # raw_input('hit enter')
                    else:
                        print('no bb found')
                        # except:
                        # e = sys.exc_info()[0]
                        #                print("could not read " + full_filename + " locally due to " + str(e) + ", returning None")
                        # logging.warning("could not read locally, returning None")


# continue  # input isn't a basestring nor a np.ndarray....so what is it?


def get_bb(img_array, use_visual_output=True, fname='filename'):
    faces = background_removal.find_face(img_array)
    print('len before '+str(len(faces)))
    faces = background_removal.combine_overlapping_rectangles(faces)
    print('len after '+str(len(faces)))
    dress_length = 12
    dress_width = 4
    if len(faces):

        orig_h, orig_w, d = img_array.shape
        head_x0 = int(np.mean([face[0] for face in faces]))
        head_y0 = int(np.mean([face[1] for face in faces]))
        w = int(np.mean([face[2] for face in faces]))
        h = int(np.mean([face[3] for face in faces]))
        dress_w = w * dress_width
        dress_y0 = head_y0+h
        dress_h = min(h * dress_length, orig_h - dress_y0 - 1)
        dress_x0 = max(0, head_x0 + w / 2 - dress_w / 2)
        dress_w = min(w * dress_width, orig_w - dress_x0 - 1)
        dress_box = [dress_x0, dress_y0, dress_w, dress_h]
        if use_visual_output == True:
            cv2.rectangle(img_array, (dress_box[0], dress_box[1]),
                          (dress_box[0] + dress_box[2], dress_box[1] + dress_box[3]),
                          GREEN, thickness=1)
            print('plotting img, dims:' + str(orig_w) + ' x ' + str(orig_h))
            # im = plt.imshow(img_array)
            # plt.show(block=False)

            cv2.imshow(fname, img_array)
            cv2.moveWindow('win', 100, 200)
            k = cv2.waitKey(200)
            raw_input('enter to continue')
            cv2.destroyAllWindows()

            if k in [27, ord('Q'), ord('q')]:  # exit on ESC
                pass
        assert (Utils.bounding_box_inside_image(img_array, dress_box))
        return dress_box
    else:
        return None


if __name__ == '__main__':
    print('starting')
    cats = ['cocktail-dresses', 'bridal-dresses', 'evening-dresses', 'day-dresses']
    keywords = ['mini', 'midi', 'maxi']

    # from scipy import misc
    # l = misc.lena()
    # import matplotlib.pyplot as plt
    # plt.imshow(l)
    # plt.show()

    dir = '/home/jeremy/jeremy.rutman@gmail.com/TrendiGuru/techdev/trendi_guru_modules/classifier_stuff/images/dresses'
    dir = 'images/dresses'
    box_images(dir, use_visual_output=True)
    raw_input('hit enter')
    read_bbs_in_subdirs(dir)
 #    get_items(['dresses'], keywords)
    # get_items(cats, None)

    # check what all the subcats are
    # db = pymongo.MongoClient().mydb
    # subcategory_id_list = find_similar_mongo.get_all_subcategories(db.categories, cat)
    # print('sub cat id list for '+str(cat)+' is:'+str(subcategory_id_list))


    # Dresses
    #    Cocktail Dresses
    #    Bridal Dresses
    ##    Evening Dresses
    #   Day Dresses

    #get_