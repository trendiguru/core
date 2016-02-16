__author__ = 'jeremy'

import os
import cv2
import hashlib
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np

def image_stats_from_dir(dirname):
    only_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    hlist = []
    wlist = []
    dlist = []
    Blist = []
    Glist = []
    Rlist = []
    n=0
    for filename in only_files:
        fullpath = os.path.join(dirname,filename)
        results = image_stats(fullpath)
        print(results)
        hlist.append(results[0])
        wlist.append(results[1])
        dlist.append(results[2])
        Blist.append(results[3])
        Glist.append(results[4])
        Rlist.append(results[5])
        n += 1
    avg_h = np.mean(hlist)
    avg_w = np.mean(wlist)
    avg_d = np.mean(dlist)
    avg_B = np.mean(Blist)
    avg_G = np.mean(Glist)
    avg_R = np.mean(Rlist)
    print('averages of {} images: h:{} w{} d{} B {} G {} R {}'.format(n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R))
    return(n,avg_h,avg_w,avg_d,avg_B,avg_G,avg_R)

def image_stats(filename):
    img_arr = cv2.imread(filename)
    if img_arr is not None:
        cv2.imshow('current_fig',img_arr)
        cv2.waitKey(10)
        shape = img_arr.shape
        if len(shape)>2:   #BGR
            h=shape[0]
            w = shape[1]
            d = shape[2]
            avgB = np.mean(img_arr[:,:,0])
            avgG = np.mean(img_arr[:,:,1])
            avgR = np.mean(img_arr[:,:,2])
            return([h,w,d,avgB,avgG,avgR])
        else:  #grayscale /single-channel image has no 3rd dim
            h=shape[0]
            w=shape[1]
            d=1
            avgGray = np.mean(img_arr[:,:])
            return([h,w,1,avgGray,avgGray,avgGray])

    else:
        logging.warning('could not open {}'.format(filename))
        return None


def remove_dupe_images(dir):
    '''
    remove dupe files from dir  - warning this deletes files
    :param dir:
    :return: number of dupes removed
    '''
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    print('n files:'+str(len(files)))
    hashes = []
    dupe_count = 0
    for a_file in files:
        fullname = os.path.join(dir,a_file)
#        img_arr = cv2.imread(fullname)
        with open(fullname,'r') as f:
            logging.debug('current file:'+fullname)
            contents = f.read()
            if contents is not None:
                m = hashlib.md5()
                m.update(contents)
                current_hash = m.hexdigest()
                logging.debug('image hash:' + current_hash + ' for ' + a_file)
                dupe_flag = False
                for a_previous_hash in hashes:
                    if  current_hash == a_previous_hash:
                        fullpath = os.path.join(dir,a_file)
                        print('going to remove '+str(fullpath))
                        os.remove(fullpath)
                        dupe_flag = True
                        dupe_count = dupe_count + 1
                        break
                if not dupe_flag:
                    hashes.append(current_hash)
                    print(fullname+' not a dupe')
    print('found {} dupes'.format(dupe_count))

if __name__ == "__main__":
    remove_dupe_images('/media/jr/Transcend/my_stuff/tg/tg_ultimate_image_db/ours/test')
#    image_stats_from_dir('/home/jr/python-packages/trendi/classifier_stuff/caffe_nns/dataset/train_pairs_belts/')