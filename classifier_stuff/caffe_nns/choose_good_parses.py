__author__ = 'jeremy'
import os
import shutil
import time
import random

from trendi.utils import imutils
from trendi.constants import fashionista_categories
from trendi import Utils

def choose_good_parses(dir):
    n_done = 0
    parse_list = [f for f in os.listdir(dir) if '.bmp' in f]
    random.shuffle(parse_list)
    img_list = []
    start_time = time.time()
    for parse in parse_list:
        full_parsename = os.path.join(dir,parse)
        img_filename = full_parsename[:-4]+'.jpg'
        if os.path.exists(img_filename):
            img_list.append(img_filename)
            c,k = imutils.show_mask_with_labels(full_parsename,labels=fashionista_categories,original_image = img_filename,visual_output=True,cut_the_crap=True)
            posename = full_parsename[:-4]+'.pose'
            move_the_rest(posename,k)
            urlname = full_parsename[:-4]+'.url'
            move_the_rest(urlname,k)
            n_done += 1
            print('n_done {} pics/minute {}'.format(n_done,(time.time()-start_time)/n_done))
        else:
            print(img_filename+' doesnt exists')

def move_the_rest(mask_filename,k):
    if not os.path.exists(mask_filename):
        print(mask_filename+' does not exist')
        return
    indir = os.path.dirname(mask_filename)
    parentdir = os.path.abspath(os.path.join(indir, os.pardir))
    curdir = os.path.split(indir)[1]  #the parent of current dir
    print('in {} parent {} cur {}'.format(indir,parentdir,curdir))
    if k == ord('d'):
        newdir = curdir+'_removed'
        dest_dir = os.path.join(parentdir,newdir)
        Utils.ensure_dir(dest_dir)
        print('REMOVING moving {} to {}'.format(mask_filename,dest_dir))
        shutil.move(mask_filename,dest_dir)

    elif k == ord('c'):
        newdir = curdir+'_needwork'
        dest_dir = os.path.join(parentdir,newdir)
        Utils.ensure_dir(dest_dir)
        print('CLOSE so moving {} to {}'.format(mask_filename,dest_dir))
        shutil.move(mask_filename,dest_dir)

    elif k == ord('k'):
        newdir = curdir+'_kept'
        dest_dir = os.path.join(parentdir,newdir)
        Utils.ensure_dir(dest_dir)
        print('KEEPING moving {} to {}'.format(mask_filename,dest_dir))
        shutil.move(mask_filename,dest_dir)

    else:
        print('doing nothing')


if __name__ == "__main__":
    dir = '/media/jeremy/5750-C34D/pd_output_test'
    dir = '/media/jeremy/5750-C34D/pd_output'
    choose_good_parses(dir)